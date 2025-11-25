%% DEMO: GPS Local Policy Update using iLQR and KL-constrained iLQR
% 
% 이 스크립트는 Guided Policy Search (GPS)의 로컬 정책 최적화 단계를 시뮬레이션합니다.
% 전체 파이프라인은 다음과 같습니다.
% 1. (가상) Roll-out 데이터 생성
% 2. 생성된 데이터로 Local Dynamics 추정 (`estimate_local_dynamics.m`)
% 3. 추정된 Dynamics를 이용하여 표준 iLQR 수행 (`run_ilqr_local_update.m`)
% 4. KL-Divergence 제약을 포함한 iLQR 수행 (`run_ilqr_with_kl_constraint.m`)
% 5. 결과 비교 및 시각화

clear; close all; clc;

%% 1. 파라미터 및 환경 설정
fprintf('=============== 1. 파라미터 설정 (learn_local_dynamics_rr.m의 값 사용) ===============\n');

% --- learn_local_dynamics_rr.m 실행 ---
% 원본 파일을 수정하지 않기 위해, 스크립트 실행을 통해 데이터를 가져옵니다.
% 이 스크립트가 nx, nu, T, N_traj, X_data, U_data 등을 워크스페이스에 생성합니다.
run('learn_local_dynamics_rr.m');

% learn_local_dynamics_rr.m에서 설정된 값들을 가져와 이 스크립트에서 사용
x_dim = nx;
u_dim = nu;
% T와 N_traj는 learn_local_dynamics_rr.m에서 정의된 것을 사용합니다.

% --- iLQR 및 KL-iLQR 옵션 ---
ilqr_opts = struct(... 
    'max_iter', 100, ...            % 최대 반복 횟수
    'cost_converge', 1e-5, ...      % 비용 수렴 한계값
    'lambda', 1e-2, ...             % 초기 Regularization 파라미터
    'lambda_factor', 1.6, ...       % Lambda 업데이트 비율
    'line_search_a', 0.5, ...       % Line search step 감소 비율
    'line_search_k', 10 ...         % Line search 최대 시도 횟수
);

kl_opts = ilqr_opts; % 기본 iLQR 옵션 상속
kl_opts.lambda_kl = 5.0; % KL 페널티 가중치 (이 값을 바꿔보세요!)
kl_opts.W_kl = eye(u_dim);  % KL 페널티 행렬

fprintf('... 설정 완료 (learn_local_dynamics_rr.m의 데이터 사용).\n');

%% 2. learn_local_dynamics_rr.m에서 가져온 Roll-out 데이터 확인
fprintf('\n=============== 2. Roll-out 데이터 확인 ===============\n');
fprintf('learn_local_dynamics_rr.m으로부터 %d개의 Roll-out 데이터 (상태: %d, 입력: %d, 시계열 길이: %d)를 가져왔습니다.\n', N_traj, x_dim, u_dim, T);



%% 3. Local Dynamics 추정
fprintf('\n=============== 3. Local Dynamics 추정 ===============\n');
[A_seq, B_seq, c_seq, ~] = estimate_local_dynamics(X_data, U_data);
fprintf('... `estimate_local_dynamics` 실행 완료.\n');


%% 4. 비용 함수 및 초기 Trajectory 설정
fprintf('\n=============== 4. 비용 함수 및 초기 Trajectory 설정 ===============\n');

x_target = [pi/4; pi/2; 0; 0]; % 목표 상태
Q = diag([10, 10, 0.1, 0.1]);
R = 0.01 * eye(u_dim);
Q_final = 100 * Q;

cost_funs = cost_functions_example(x_target, Q, R, Q_final);
fprintf('... 비용 함수 정의 완료.\n');

% 초기 Trajectory: 평균 궤적 사용
x_traj_init = mean(X_data, 3);
u_traj_init = mean(U_data, 3);
fprintf('... 초기 Trajectory를 평균 궤적으로 설정 완료.\n');


%% 5. 표준 iLQR 실행
fprintf('\n=============== 5. 표준 iLQR 실행 ===============\n');

[K_iLQR, k_iLQR, x_iLQR, u_iLQR, cost_iLQR] = run_ilqr_local_update(...
    x_traj_init, u_traj_init, A_seq, B_seq, c_seq, cost_funs, ilqr_opts);


%% 6. KL 제약을 포함한 iLQR 실행
fprintf('\n=============== 6. KL 제약 포함 iLQR 실행 ===============\n');

% Reference action sequence: 여기서는 간단히 초기(평균) 입력을 사용.
% 실제 GPS에서는 이전 글로벌 정책의 출력을 사용합니다.
u_ref_seq = u_traj_init;

[K_KL, k_KL, x_KL, u_KL, cost_KL, kl_div] = run_ilqr_with_kl_constraint(...
    x_traj_init, u_traj_init, A_seq, B_seq, c_seq, cost_funs, u_ref_seq, kl_opts);


%% 7. 결과 시각화 및 비교
fprintf('\n=============== 7. 결과 시각화 ===============\n');

% --- 비용 감소 과정 ---
% [학습 포인트 1]
% 이 플롯은 표준 iLQR과 KL 제약 iLQR의 반복에 따른 총 비용 감소를 보여줍니다.
% - 두 그래프 모두 반복이 진행됨에 따라 비용이 감소하고 수렴하는 것을 통해, 
%   최적화가 성공적으로 이루어졌음을 확인할 수 있습니다.
% - KL 제약 iLQR(빨간색)의 최종 비용이 더 높은 것은, 순수한 비용 최소화뿐만 아니라
%   KL 페널티(기준 정책과의 유사성 유지)까지 고려하는 트레이드오프가 있기 때문입니다.
figure('Name', 'Cost Convergence');
plot(cost_iLQR, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'Standard iLQR');
hold on;
plot(cost_KL, 'r-s', 'LineWidth', 1.5, 'DisplayName', 'KL-constrained iLQR');
grid on;
xlabel('Iteration');
ylabel('Total Cost');
title('Cost Convergence Comparison');
legend;
set(gca, 'FontSize', 12);

% --- 상태 Trajectory 비교 ---
% [학습 포인트 2]
% 이 플롯은 각 상태 변수(q1, q2, dq1, dq2)의 시간에 따른 변화를 보여줍니다.
% - 초기 평균 궤적(검은색 점선)에서 시작하여, 두 iLQR 알고리즘이 성공적으로 
%   시스템을 목표 상태(녹색 별)로 유도하는 것을 시각적으로 확인할 수 있습니다.
% - 표준 iLQR과 KL-iLQR이 목표에 도달하기 위해 약간 다른 경로를 생성할 수 있음을
%   비교해 볼 수 있습니다.
figure('Name', 'State Trajectories');
state_names = {'q1', 'q2', 'dq1', 'dq2'};
for i = 1:x_dim
    subplot(x_dim, 1, i);
    plot(0:T, x_traj_init(i,:), 'k:', 'LineWidth', 1, 'DisplayName', 'Initial (mean)');
    hold on;
    plot(0:T, x_iLQR(i,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'iLQR');
    plot(0:T, x_KL(i,:), 'r--', 'LineWidth', 1.5, 'DisplayName', 'KL-iLQR');
    plot(T, x_target(i), 'g*', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Target');
    ylabel(state_names{i});
    grid on;
    if i == 1
        title('State Trajectories Comparison');
        legend('Location', 'northeast');
    end
    if i == x_dim
        xlabel('Time step (t)');
    end
    set(gca, 'FontSize', 12);
end

% --- 입력 Trajectory 비교 ---
% [학습 포인트 3: KL 제약의 핵심 효과]
% 이 플롯은 KL 제약의 효과를 가장 명확하게 보여줍니다.
% - 표준 iLQR(파란색)은 비용 최소화만을 위해 자유롭게 제어 입력을 찾습니다.
% - KL 제약 iLQR(빨간색 점선)은 기준 입력(검은색 점선)에서 크게 벗어나지 않으려는 
%   경향을 보입니다. 즉, KL 페널티 항에 의해 제어 입력이 기준 입력 쪽으로 "끌어당겨지는"
%   효과가 나타납니다.
% - 이는 GPS에서 로컬 정책이 글로벌 정책에서 너무 급격하게 변하는 것을 막아
%   학습 안정성을 높이는 '신뢰 영역(Trust Region)'의 역할을 합니다.
figure('Name', 'Control Input Trajectories');
for i=1:u_dim
    subplot(u_dim, 1, i);
    plot(1:T, u_iLQR(i,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'iLQR');
    hold on;
    plot(1:T, u_KL(i,:), 'r--', 'LineWidth', 1.5, 'DisplayName', 'KL-iLQR');
    plot(1:T, u_ref_seq(i, :), 'k:', 'LineWidth', 1, 'DisplayName', 'Reference (mean)');
    ylabel(sprintf('u%d', i));
    grid on;
    if i == 1
        title('Control Input Comparison');
        legend;
    end
    if i == u_dim
        xlabel('Time step (t)');
    end
    set(gca, 'FontSize', 12);
end

% --- KL-iLQR의 KL-Divergence 변화 ---
% [학습 포인트 4]
% 이 플롯은 KL 제약 iLQR의 각 반복에서 계산된 KL 다이버전스(근사값)를 보여줍니다.
% - 이 값은 현재 컨트롤러가 기준 컨트롤러와 얼마나 다른지를 나타내는 척도입니다.
% - 최적화 과정에서 알고리즘이 비용 감소와 KL 페널티 사이에서 어떻게 균형을
%   맞추는지 확인할 수 있습니다.
figure('Name', 'KL Divergence Trace');
plot(kl_div, 'm-d', 'LineWidth', 1.5);
grid on;
xlabel('Iteration');
ylabel('Approx. KL Divergence');
title('KL-Divergence Trace for KL-constrained iLQR');
set(gca, 'FontSize', 12);

fprintf('... 시각화 완료.\n');
