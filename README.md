# dVRK Force Estimation
Using neural network to estimate forces on the dVRK. Start with preprocessing to convert from Rosbag to CSV files for training the networks on. Use either the direct or the indirect method for force estimation. The indirect method includes an extension to account for trocar interaction froces. The FSR section is a work in progress to include grip force estimation. Each section is further detailed by the README in its directory. 

We test two methods here 
 * Indirect - which learns the torque to move the joints in free space as described in this paper https://www.researchgate.net/publication/341879597_Neural_Network_based_Inverse_Dynamics_Identification_and_External_Force_Estimation_on_the_da_Vinci_Research_Kit
 * Direct - which learns the forces directly from a sensor in the environment as descriped in this paper https://ieeexplore.ieee.org/abstract/document/9287941

## This branch is about hybrid, model based (convex opt-based dynamics ID) plus model free (MLP based trocar correction).
Details see ReadMe in the daVinci_system_id folder.

## How to Install

### Clone the Repository

Firstly, git clone this repository with recursive submodule:

```bash
git clone --recursive https://github.com/vu-maple-lab/dvrk_force_estimation.git
```

Please make sure you have the option `--recursive` since it is required to clone the submodule as well.

Then, head to the path of the cloned repository for the next step.

```bash
cd <dvrk_force_estimation>
```

### Install pyOpt

```bash
cd external_package
cd pyOpt
python setup.py install
```
Note: in case pyOpt folder is empty, the recursive cloning has failed. You will need to update this submodule mannually. Try:

```bash
git submodule update --recursive
```

Once pyOpt has the repo pulled, you can then install the submodule:

```bash
python setup.py install
```

Next, head to the path of the cloned repository for the next step.

```bash
cd <dvrk_force_estimation>
```

### Install local package `dvrk_dynamic_identification`

```bash
cd dynamic_identification
pip3 install -e .
```
