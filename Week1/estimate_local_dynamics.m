function [A_seq, B_seq, c_seq, Sigma_seq] = estimate_local_dynamics(X_data, U_data)
%estimate_local_dynamics - Roll-out 데이터로부터 Local Dynamics 시퀀스를 추정합니다.
% 
% 이 함수는 learn_local_dynamics_rr.m 스크립트의 3번 섹션(선형 회귀) 로직을
% 일반화하여, 모든 시간 스텝 t = 1...T 에 대해 Local Dynamics를 계산합니다.
% 
% 입력:
%   X_data (n x T+1 x N) - N개의 Roll-out에 대한 상태 trajectory 데이터
%   U_data (m x T x N)   - N개의 Roll-out에 대한 입력 trajectory 데이터
% 
% 출력:
%   A_seq (cell array, T) - 각 시점 t의 상태 Jacobian (A_t)
%   B_seq (cell array, T) - 각 시점 t의 입력 Jacobian (B_t)
%   c_seq (cell array, T) - 각 시점 t의 상수항 (c_t)
%   Sigma_seq (cell array, T) - 각 시점 t의 노이즈 공분산 (Sigma_t)
% 

% 데이터 차원 가져오기
[nx, T_plus_1, N_traj] = size(X_data);
nu = size(U_data, 1);
T = size(U_data, 2);
assert(T_plus_1 == T + 1, 'X_data should have length T+1');

% 출력 변수 초기화
A_seq = cell(1, T);
B_seq = cell(1, T);
c_seq = cell(1, T);
Sigma_seq = cell(1, T);

fprintf('Running local dynamics estimation for t = 1...%d\n', T);

% 모든 시간 스텝에 대해 루프 실행
for t = 1:T
    % 현재 시간 스텝 t에서의 데이터 추출
    % squeeze 함수는 (nx, 1, N_traj) 같은 차원을 (nx, N_traj)로 변경
    Xt = squeeze(X_data(:, t, :));
    Ut = squeeze(U_data(:, t, :));
    X_next = squeeze(X_data(:, t+1, :));
    
    % 선형 회귀를 위한 입력 벡터 구성: z_i = [x_i; u_i; 1]
    % Z 행렬의 크기: (nx+nu+1, N_traj)
    Z = [Xt; Ut; ones(1, N_traj)];
    
    % 선형 회귀로 파라미터 [A_t, B_t, c_t] 추정
    % X_next = [A_t, B_t, c_t] * Z
    % MATLAB의 `\` 연산자를 사용하여 안정적으로 해를 구함
    % (A_t*Z)\' = Z\' * A_t\'  =>  A_t\' = Z\' \ X_next\'
    params_T = Z' \ X_next';
    params = params_T'; % 크기: [nx x (nx + nu + 1)]
    
    % 추정된 파라미터를 A_t, B_t, c_t로 분해
    A_seq{t} = params(:, 1:nx);
    B_seq{t} = params(:, nx+1:nx+nu);
    c_seq{t} = params(:, nx+nu+1);
    
    % 노이즈 공분산 추정
    % 잔차(residual) 계산: epsilon = x_{t+1} - (A_t*x_t + B_t*u_t + c_t)
    X_pred = A_seq{t} * Xt + B_seq{t} * Ut + c_seq{t} * ones(1, N_traj);
    residuals = X_next - X_pred;
    
    % 잔차의 공분산 행렬 계산
    % cov 함수는 변수가 행에 오도록 입력을 받으므로, residuals\'를 사용
    Sigma_seq{t} = cov(residuals');
end

fprintf('... Local dynamics estimation finished.\n');

end
