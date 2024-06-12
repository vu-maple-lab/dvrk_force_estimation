import sympy
import numpy as np
from dvrk_dynamic_identification.utils import vec2so3
import copy
from dvrk_dynamic_identification.dynamics.dyn_param_dep import find_dyn_parm_deps

class Dynamics:
    def __init__(self, rbt_def, geom, g=[0, 0, -9.81], verbose=False):
        self.rbt_def = rbt_def
        self.geom = geom
        self._g = np.matrix(g)

        self._calc_dyn()
        self._calc_regressor()
        self._calc_MCG()
        self._calc_base_param()

        print("Finished creating robot dynamics")

    def _ml2r(self, m, l):
        return sympy.Matrix(l) / m

    def _Lmr2I(self, L, m, r):
        return sympy.Matrix(L - m * vec2so3(r).transpose() * vec2so3(r))

    def _calc_dyn(self):
        print("Calculating Lagrangian...")
        # Calculate kinetic energy and potential energy
        p_e = 0
        k_e = 0
        self.k_e3 = 0

        for num in self.rbt_def.link_nums[1:]:
            k_e_n = 0
            if self.rbt_def.use_inertia[num]:
                print("Calculating the link kinetic energy of {}/{}".format(num, self.rbt_def.link_nums[-1]))
                p_e += -self.rbt_def.m[num] * self.geom.p_c[num].dot(self._g)

                k_e_n = self.rbt_def.m[num] * self.geom.v_cw[num].dot(self.geom.v_cw[num])/2 +\
                       (self.geom.w_b[num].transpose() * self.rbt_def.I_by_Llm[num] * self.geom.w_b[num])[0, 0]/2

                # k_e_n = sympy.simplify(k_e_n) # this is replaced by the following code to reduce time cost
                k_e_n = sympy.factor(sympy.expand(k_e_n) - sympy.expand(k_e_n * self.rbt_def.m[num]).subs(self.rbt_def.m[num], 0)/self.rbt_def.m[num])

            k_e += k_e_n

        # Lagrangian
        L = k_e - p_e

        tau = []

        print("Calculating joint torques...")
        for q, dq in zip(self.rbt_def.coordinates, self.rbt_def.d_coordinates):
            print("tau of {}".format(q))
            dk_ddq = sympy.diff(k_e, dq)
            dk_ddq_t = dk_ddq.subs(self.rbt_def.subs_q2qt + self.rbt_def.subs_dq2dqt)
            dk_ddq_dtt = sympy.diff(dk_ddq_t, sympy.Symbol('t'))

            dk_ddq_dt = dk_ddq_dtt.subs(self.rbt_def.subs_ddqt2ddq + self.rbt_def.subs_dqt2dq + self.rbt_def.subs_qt2q)

            dL_dq = sympy.diff(L, q)

            tau.append(sympy.expand(dk_ddq_dt - dL_dq))

        print("Adding frictions and springs...")
        tau = copy.deepcopy(tau)

        for i in range(self.rbt_def.frame_num):
            dq = self.rbt_def.dq_for_frame[i]

            if self.rbt_def.use_friction[i]:
                tau_f = sympy.sign(dq) * self.rbt_def.Fc[i] + dq * self.rbt_def.Fv[i] + self.rbt_def.Fo[i]
                for a in range(len(self.rbt_def.d_coordinates)):
                    dq_da = sympy.diff(dq, self.rbt_def.d_coordinates[a])
                    tau[a] += dq_da * tau_f

            if self.rbt_def.spring_dl[i] != None:
                tau_s = self.rbt_def.spring_dl[i] * self.rbt_def.K[i]
                for a in range(len(self.rbt_def.d_coordinates)):
                    dq_da = sympy.diff(dq, self.rbt_def.d_coordinates[a])
                    tau[a] -= dq_da * tau_s

        # for k in range(len(self.rbt_def.K)):
        #     tau_k = self.rbt_def.springs[k] * self.rbt_def.K[k]
        #     index = self.rbt_def.coordinates.index(list(self.rbt_def.springs[k].free_symbols)[0])
        #     tau[index] += -tau_k

        print("Add motor inertia...")

        for i in range(self.rbt_def.frame_num):
            if self.rbt_def.use_Ia[i]:
                tau_Ia = self.rbt_def.ddq_for_frame[i] * self.rbt_def.Ia[i]
                tau_index = self.rbt_def.dd_coordinates.index(self.rbt_def.ddq_for_frame[i].free_symbols.pop())
                tau[tau_index] += tau_Ia

        # for tendon_coupling in self.rbt_def.tendon_couplings:
        #     src_frame, dst_frame, k = tendon_coupling
        #     dq_src = self.rbt_def.dq_for_frame[src_frame]
        #     dq_dst = self.rbt_def.dq_for_frame[dst_frame]
        #     src_index = self.rbt_def.d_coordinates.index(dq_src)
        #
        #     for a in range(len(self.rbt_def.d_coordinates)):
        #         dq_da = sympy.diff(dq_dst, self.rbt_def.d_coordinates[a])
        #         tau_c[a] += dq_da * k * tau_csf[src_index]

        self.tau = tau

    def _calc_regressor(self):
        print("Calculating regressor...")
        A, b = sympy.linear_eq_to_matrix(self.tau, self.rbt_def.bary_params)

        self.H = A

        input_vars = tuple(self.rbt_def.coordinates + self.rbt_def.d_coordinates + self.rbt_def.dd_coordinates)
        # print('input_vars', input_vars)
        self.H_func = sympy.lambdify(input_vars, self.H)

    def _calc_base_param(self):
        print("Calculating base parameter...")
        r, P_X, P = find_dyn_parm_deps(len(self.rbt_def.coordinates), len(self.rbt_def.bary_params), self.H_func)
        self.base_num = r
        print('base parameter number: {}'.format(self.base_num))
        self.base_param = P_X.dot(np.matrix(self.rbt_def.bary_params).transpose())

        P_b = P[:r].tolist()


        self.H_b = self.H[:, P_b]

        input_vars = tuple(self.rbt_def.coordinates + self.rbt_def.d_coordinates + self.rbt_def.dd_coordinates)

        print("Creating H_b function...")
        self.H_b_func = sympy.lambdify(input_vars, self.H_b)

    def _calc_M(self):
        A, b = sympy.linear_eq_to_matrix(self.tau, self.rbt_def.dd_coordinates)

        self.M = A

    def _calc_G(self):
        subs_qdq2zero = [(dq, 0) for dq in self.rbt_def.d_coordinates]
        subs_qdq2zero += [(ddq, 0) for ddq in self.rbt_def.dd_coordinates]
        self.G = sympy.Matrix(self.tau).subs(subs_qdq2zero)


    def _calc_C(self):
        subs_ddq2zero = [(ddq, 0) for ddq in self.rbt_def.dd_coordinates]
        self.C = sympy.Matrix(self.tau).subs(subs_ddq2zero) - self.G

    def _calc_MCG(self):
        print("Calculating M, C and G...")
        self._calc_M()
        self._calc_G()
        self._calc_C()
