data = 'free_space';
contact = 'no_contact';
test_folder = 'test';
rnn = 'lstm';
network = '_seal_pred_filtered_torque_si.csv';

%loss = 0;
loss = [0,0,0,0];

%for file = 0:3
file = 0;
exp = ['exp',num2str(file)];

if strcmp(test_folder, 'test')
%     joint_path = ['../../data_2_23/csv_si/', test_folder, '/', data, '/', contact, '/', exp, '/joints/'];
%     torque_path = ['../../data_2_23/csv_si/', test_folder, '/', data, '/', contact, '/', exp, '/', rnn, network];
      joint_path = ['../../data_2_23/csv_si/', test_folder, '/', data, '/joints/'];
      torque_path = ['../../data_2_23/csv_si/', test_folder, '/', data, '/', 'lstm', network];
else
    joint_path = ['../../data_2_23/csv_si/', test_folder, '/', data, '/joints/'];
    torque_path = ['../../data_2_23/csv_si/', test_folder, '/', data, '/', 'lstm', network];
end

joint_data = readmatrix([joint_path, 'interpolated_all_joints.csv']);
torque_data = readmatrix(torque_path);

measured_torque = joint_data(:,14:16);
fs_pred_torque = torque_data(:,2:4);
loss(file+1) = mean(sqrt(mean((measured_torque(1:length(fs_pred_torque),:) - fs_pred_torque).^2)));

figure()
tcl = tiledlayout(2,3);

title(tcl, sprintf('TRANSFORMER-Si %s: RSME = %.4f', test_folder, loss(1)))


nexttile
plot(joint_data(:, 1)-joint_data(1,1), joint_data(:,14), 'b')
hold on
title('joint 1')
plot(torque_data(:, 1), torque_data(:,2), 'r')
legend('measured', 'predicted')
hold off

nexttile
plot(joint_data(:, 1)-joint_data(1,1), joint_data(:,15), 'b')
hold on
title('joint 2')
plot(torque_data(:, 1), torque_data(:,3), 'r')
legend('measured', 'predicted')

nexttile
plot(joint_data(:, 1)-joint_data(1,1), joint_data(:,16), 'b')
hold on
title('joint 3')
plot(torque_data(:, 1), torque_data(:,4), 'r')
legend('measured', 'predicted')

nexttile
plot(joint_data(:, 1)-joint_data(1,1), joint_data(:,17), 'b')
hold on
title('joint 4')
plot(torque_data(:, 1), torque_data(:,5), 'r')
legend('measured', 'predicted')

nexttile
plot(joint_data(:, 1)-joint_data(1,1), joint_data(:,18), 'b')
hold on
title('joint 5')
plot(torque_data(:, 1), torque_data(:,6), 'r')
legend('measured', 'predicted')

nexttile
plot(joint_data(:, 1)-joint_data(1,1), joint_data(:,19), 'b')
hold on
title('joint 6')
plot(torque_data(:, 1), torque_data(:,7), 'r')
legend('measured', 'predicted')



% subplot(2,3,1)
% plot(joint_data(:, 1)-joint_data(1,1), joint_data(:,14), 'b')
% title(data)
% hold on
% plot(torque_data(:, 1), torque_data(:,2), 'r')
% legend('measured', 'predicted')
% title(data)
% hold off
% subplot(2,3,2)
% plot(joint_data(:, 1)-joint_data(1,1), joint_data(:,15), 'b')
% title(data)
% hold on
% plot(torque_data(:, 1), torque_data(:,3), 'r')
% legend('measured', 'predicted')
% title('Torque')
% hold off
% subplot(2,3,3)
% plot(joint_data(:, 1)-joint_data(1,1), joint_data(:,16), 'b')
% title(data)
% hold on
% plot(torque_data(:, 1), torque_data(:,4), 'r')
% legend('measured', 'predicted')
% title('Torque')
% hold off
% subplot(2,3,4)
% plot(joint_data(:, 1)-joint_data(1,1), joint_data(:,17), 'b')
% title(data)
% hold on
% plot(torque_data(:, 1), torque_data(:,5), 'r')
% legend('measured', 'predicted')
% title('Torque')
% hold off
% subplot(2,3,5)
% plot(joint_data(:, 1)-joint_data(1,1), joint_data(:,18), 'b')
% title(data)
% hold on
% plot(torque_data(:, 1), torque_data(:,6), 'r')
% legend('measured', 'predicted')
% title('Torque')
% hold off
% subplot(2,3,6)
% plot(joint_data(:, 1)-joint_data(1,1), joint_data(:,19), 'b')
% title(data)
% hold on
% plot(torque_data(:, 1), torque_data(:,7), 'r')
% legend('measured', 'predicted')
% title('Torque')
% hold off
%end
