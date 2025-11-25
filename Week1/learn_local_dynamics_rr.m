%% 2-DOF RR 로봇팔의 Local Dynamics 추정 스크립트
% 이 스크립트는 Levine et al. (2016) "End-to-End Training of Deep Visuomotor Policies"에서
% 사용된 방법론과 유사하게, roll-out 데이터를 통해 로봇의 지역적 동역학(local dynamics)을
% 선형 회귀로 추정하는 과정을 시뮬레이션합니다.

clear; clc; close all;

%% 1. Setup: 2-DOF RR robot definition
% 이 블록은 시뮬레이션에 사용할 2-DOF RR 로봇팔의 물리적 파라미터와
% 동역학 모델(continuous & discrete)을 정의합니다.
% 논문에서의 실제 로봇(physical system)에 해당합니다.

disp('1. 로봇 모델 및 시뮬레이션 설정 중...');

% 물리 파라미터 정의
param.m1 = 1.0; % 링크 1 질량 (kg)
param.m2 = 1.0; % 링크 2 질량 (kg)
param.l1 = 1.0; % 링크 1 길이 (m)
param.l2 = 1.0; % 링크 2 길이 (m)
param.lc1 = 0.5; % 링크 1 질량중심까지의 거리 (m)
param.lc2 = 0.5; % 링크 2 질량중심까지의 거리 (m)
param.I1 = 0.1; % 링크 1 관성 모멘트 (kg*m^2)
param.I2 = 0.1; % 링크 2 관성 모멘트 (kg*m^2)
param.g = 9.81; % 중력 가속도 (m/s^2)

% 상태 및 입력 차원
nx = 4; % 상태 벡터 x = [q1, q2, dq1, dq2]의 차원
nu = 2; % 입력 벡터 u = [tau1, tau2]의 차원

% 시뮬레이션 시간 설정
dt = 0.01; % 이산화 시간 간격 (초)

disp('   ... 완료.');

%% 2. Roll-out: collect (x_t, u_t, x_{t+1}) samples
% 이 블록은 "model-free roll-out"을 수행하여 동역학 학습에 필요한 데이터를 수집합니다.
% 초기 정책(여기서는 PD 컨트롤러)을 사용하여 여러 궤적(trajectory)을 생성하고,
% 각 시간 스텝에서 (현재 상태, 현재 입력, 다음 상태) 튜플을 기록합니다.
% 이는 trajectory-centric RL에서 policy search를 위한 데이터를 모으는 과정과 같습니다.
% 각 에피소드는 랜덤초기값에서 시간에 따라 목표점에 도달하는 여러가지 경로를 보여줌

disp('2. Roll-out을 통해 학습 데이터 수집 중...');

% Roll-out 설정
T = 100;       % 총 시뮬레이션 스텝 수
N_traj = 100;  % 총 roll-out(궤적) 개수

% 데이터 저장용 변수 초기화
% X_data: (nx, T+1, N_traj) 크기의 3차원 배열. 각 궤적의 상태(x) 시퀀스 저장
% U_data: (nu, T, N_traj) 크기의 3차원 배열. 각 궤적의 입력(u) 시퀀스 저장
X_data = zeros(nx, T + 1, N_traj);
U_data = zeros(nu, T, N_traj);

% PD 컨트롤러 설정
q_des = [pi/4; pi/2]; % 목표 각도 (rad)
Kp = 50;             % Proportional gain
Kd = 10;             % Derivative gain

% Roll-out 루프
for k = 1:N_traj
    % 각 roll-out마다 초기 상태에 랜덤 노이즈 추가
    x0 = [0; 0; 0; 0] + 0.1 * randn(nx, 1);
    X_data(:, 1, k) = x0;
    
    x_t = x0;
    
    % 안쪽 루프: 시간 스텝 진행
    for t = 1:T
        % 현재 상태 분리
        q = x_t(1:2);
        dq = x_t(3:4);
        
        % 1. PD 컨트롤러로 입력 u_t 계산
        % 이 부분이 논문에서의 초기 컨트롤러(policy)에 해당합니다.
        % 이 컨트롤러를 실행하여 얻은 데이터로 더 나은 컨트롤러를 학습합니다.
        u_t = Kp * (q_des - q) - Kd * dq;
        
        % 2. Dynamics step으로 x_{t+1} 계산
        x_next = step_dynamics(x_t, u_t, dt, param);
        
        % 3. (x_t, u_t, x_{t+1}) 데이터 저장
        % trajectory-centric RL에서는 이 튜플이 한 번의 '경험'에 해당하며,
        % 이 경험들을 모아 모델을 학습하거나 가치 함수를 업데이트합니다.
        X_data(:, t+1, k) = x_next;
        U_data(:, t, k) = u_t;
        
        % 다음 스텝을 위해 상태 업데이트
        x_t = x_next;
    end
end

disp('   ... 완료.');

%% 3. Local dynamics estimation via linear regression
% 이 블록은 수집된 roll-out 데이터 중 특정 시점(t0)의 샘플들을 사용하여
% 그 시점 근방의 지역적 동역학(local dynamics)을 선형 모델로 근사합니다.
% 이는 복잡한 전체 동역학을 알 필요 없이, 현재 궤적 주변에서만 유효한
% 간단한 선형-가우시안 모델 p(x_{t+1} | x_t, u_t)를 데이터로부터 구하는 과정입니다.
% 이렇게 추정된 (fx, fu, F_t)는 iLQR/LQR 컨트롤러를 계산하는 데 사용됩니다.

disp('3. 특정 시점(t0)에서의 Local Dynamics를 선형 회귀로 추정 중...');

% 1. 특정 time step 선택
t0 = 20; % 분석할 특정 시간 스텝

% t0에서의 모든 궤적 데이터를 2D 행렬로 추출 (에피소드들의 특정시간에 대한 데이터 추출)
% X_t0: (nx, N_traj), U_t0: (nu, N_traj), X_next_t0: (nx, N_traj)
X_t0 = squeeze(X_data(:, t0, :));
U_t0 = squeeze(U_data(:, t0, :));
X_next_t0 = squeeze(X_data(:, t0+1, :));

% 2. 회귀용 입력 벡터 구성
% x_{t+1} ≈ A_t * [x_t; u_t; 1] 형태의 선형 관계를 가정합니다.
% Z: (nx+nu+1, N_traj) 크기의 회귀 입력 행렬 Z = [x u 1]
Z = [X_t0; U_t0; ones(1, N_traj)];

% 3. 선형 회귀로 A_t 추정
% X_next_t0 = A_t * Z  =>  A_t = X_next_t0 * Z' * (Z*Z')^{-1}
% MATLAB의 `\` 연산자(mldivide)를 사용하면 더 안정적이고 효율적으로 계산 가능합니다.
% (A_t * Z)' = Z' * A_t'  =>  A_t' = Z' \ X_next_t0'
A_t_T = Z' \ X_next_t0';
A_t = A_t_T'; % A_t 크기: [nx x (nx + nu + 1)]

% A_t를 Jacobian과 상수항으로 분해
fx = A_t(:, 1:nx);             % 상태에 대한 Jacobian 근사
fu = A_t(:, nx+1:nx+nu);       % 입력에 대한 Jacobian 근사
fc = A_t(:, nx+nu+1);          % 상수항 (offset)

% 4. 노이즈 공분산 F_t 추정
% 잔차(residual) 계산: epsilon = x_{t+1} - (fx*x_t + fu*u_t + fc)
X_pred_t0 = fx * X_t0 + fu * U_t0 + fc * ones(1, N_traj);
residuals = X_next_t0 - X_pred_t0;

% 잔차의 공분산 행렬 계산
% cov 함수는 변수가 행에 오도록 입력을 받으므로, residuals를 전치해야 함.
F_t = cov(residuals'); % F_t 크기: [nx x nx]

disp('   ... 완료. 추정된 local dynamics 행렬:');
disp('fx (state Jacobian):'); disp(fx);
disp('fu (input Jacobian):'); disp(fu);
disp('fc (constant offset):'); disp(fc);
disp('F_t (noise covariance):'); disp(F_t);


%% 4. Sanity check: compare true vs. predicted x_{t+1}
% 추정된 local linear dynamics가 얼마나 정확한지 확인하는 과정입니다.
% 회귀에 사용된 t0 시점의 데이터를 이용해 예측값(x_pred)을 만들고,
% 실제 다음 상태(x_{t+1})와 비교하여 오차를 시각화합니다.
% 이를 통해 복잡한 비선형 동역학이 특정 지점 근방에서는 선형 모델로
% 잘 근사되는지 직관적으로 확인할 수 있습니다.

disp('4. Sanity Check: 실제 x_{t+1}와 예측 x_{t+1} 비교...');

% 예측 오차 계산 (L2 norm)
prediction_errors = vecnorm(X_next_t0 - X_pred_t0, 2, 1);

% 결과 시각화
figure;
set(gcf, 'Name', 'Sanity Check: Prediction vs. True Dynamics');

% 1. 예측 오차 플롯
subplot(2, 3, 1);
plot(1:N_traj, prediction_errors, 'o-');
title(sprintf('Prediction Error Norm at t=%d', t0));
xlabel('Trajectory Index');
ylabel('||x_{t+1} - x_{pred}||_2');
grid on;
axis tight;

% 2. 각 상태 변수에 대한 실제값 vs. 예측값 scatter plot
state_names = {'q1', 'q2', 'dq1', 'dq2'};
for i = 1:nx
    subplot(2, 3, i + 2);
    scatter(X_next_t0(i, :), X_pred_t0(i, :), 15, 'filled');
    hold on;
    grid on;
    
    % y=x 라인과 축을 동일하게 설정
    current_xlim = xlim;
    current_ylim = ylim;
    min_limit = min([current_xlim, current_ylim]);
    max_limit = max([current_xlim, current_ylim]);
    
    % y=x 라인 플롯
    plot([min_limit, max_limit], [min_limit, max_limit], 'r--', 'LineWidth', 1.5);
    
    % 축 범위 설정
    axis([min_limit, max_limit, min_limit, max_limit]);
    axis equal; % Ensure aspect ratio is 1:1
    
    title(sprintf('True vs. Predicted %s at t=%d', state_names{i}, t0));
    xlabel(['True x_{t+1}(' num2str(i) ')']);
    ylabel(['Predicted x_{t+1}(' num2str(i) ')']);
end

% subplot 위치 조정
h = findobj(gcf,'type','subplot');
if ~isempty(h)
    set(h,'Position',get(h,'Position').*[1 1 0.9 0.9])
end

disp('   ... 시각화 완료. 플롯을 확인하세요.');


%% Helper Functions

function dx = rr_dynamics(x, u, param)
    % 2-DOF RR 로봇팔의 연속 시간 동역학(continuous-time dynamics) 함수
    % 입력:
    %   x: 상태 벡터 [q1; q2; dq1; dq2] (4x1)
    %   u: 입력 토크 벡터 [tau1; tau2] (2x1)
    %   param: 물리 파라미터 구조체
    % 출력:
    %   dx: 상태의 시간 미분 [dq1; dq2; ddq1; ddq2] (4x1)

    % 파라미터 언패킹
    m1 = param.m1; m2 = param.m2; l1 = param.l1; l2 = param.l2;
    lc1 = param.lc1; lc2 = param.lc2; I1 = param.I1; I2 = param.I2;
    g = param.g;

    % 상태 언패킹
    q1 = x(1); q2 = x(2);
    dq1 = x(3); dq2 = x(4);

    % 동역학 행렬 계산 (교과서 공식 기반)
    % M(q) - 질량 행렬
    m11 = m1*lc1^2 + m2*(l1^2 + lc2^2 + 2*l1*lc2*cos(q2)) + I1 + I2;
    m12 = m2*(lc2^2 + l1*lc2*cos(q2)) + I2;
    m21 = m12;
    m22 = m2*lc2^2 + I2;
    M = [m11, m12; m21, m22];

    % C(q, dq) - 코리올리/원심력 행렬
    h = -m2*l1*lc2*sin(q2);
    c11 = h*dq2;
    c12 = h*(dq1 + dq2);
    c21 = -h*dq1;
    c22 = 0;
    C = [c11, c12; c21, c22];

    % G(q) - 중력 벡터
    g1 = (m1*lc1 + m2*l1)*g*cos(q1) + m2*lc2*g*cos(q1+q2);
    g2 = m2*lc2*g*cos(q1+q2);
    G = [g1; g2];

    % ddq 계산: ddq = M^{-1} * (u - C*dq - G)
    q = [q1; q2];
    dq = [dq1; dq2];
    ddq = M \ (u - C*dq - G);

    % dx = [dq; ddq] 반환
    dx = [dq; ddq];
end

function x_next = step_dynamics(x, u, dt, param)
    % Euler integration을 사용한 이산 시간 동역학 스텝 함수
    % 입력:
    %   x: 현재 상태 벡터 (nx x 1)
    %   u: 현재 입력 벡터 (nu x 1)
    %   dt: 시간 간격
    %   param: 물리 파라미터
    % 출력:
    %   x_next: 다음 상태 벡터 (nx x 1)
    
    % 연속 시간 동역학 계산
    dx = rr_dynamics(x, u, param);
    
    % 프로세스 노이즈 정의 (평균 0, 작은 표준편차)
    % 상태의 각 차원(q1, q2, dq1, dq2)에 미세한 노이즈를 추가합니다.
    process_noise_std = [0.001; 0.001; 0.01; 0.01]; % 기존보다 10배 큰 노이즈
    noise = process_noise_std .* randn(4, 1);

    % Euler integration에 프로세스 노이즈 추가
    x_next = x + dt * dx + noise;
end
