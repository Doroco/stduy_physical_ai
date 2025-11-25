function [K_seq, k_seq, x_traj, u_traj, cost_trace] = run_ilqr_local_update(x_traj_init, u_traj_init, A_seq, B_seq, c_seq, cost_funs, ilqr_opts)
%run_ilqr_local_update - iLQR을 이용한 로컬 컨트롤러 최적화
% 
% 주어진 Local Time-Varying Linear-Gaussian Dynamics 하에서
% iLQR 알고리즘을 수행하여 최적의 Time-Varying Linear Controller를 찾고,
% 새로운 nominal trajectory를 계산합니다.
% 
% 입력:
%   x_traj_init (n x T+1) - 초기 상태 trajectory guess
%   u_traj_init (m x T)   - 초기 입력 trajectory guess
%   A_seq (cell array, T) - 각 시점 t의 A_t (n x n) 행렬
%   B_seq (cell array, T) - 각 시점 t의 B_t (n x m) 행렬
%   c_seq (cell array, T) - 각 시점 t의 c_t (n x 1) 벡터 (affine term)
%   cost_funs (struct)    - 비용 함수 핸들 구조체 (cost_functions_example.m 참고)
%   ilqr_opts (struct)    - iLQR 알고리즘 옵션:
%     .max_iter       - 최대 반복 횟수
%     .cost_converge  - 비용 수렴 한계값
%     .lambda         - Q_uu regularization을 위한 초기 lambda 값
%     .lambda_factor  - lambda 증가/감소 비율
%     .line_search_a  - Line search 감소 비율 (e.g., 0.5)
%     .line_search_k  - Line search 최대 시도 횟수
% 
% 출력:
%   K_seq (cell array, T) - 최적 피드백 이득 K_t (m x n)
%   k_seq (cell array, T) - 최적 피드포워드 항 k_t (m x 1)
%   x_traj (n x T+1)      - 최적화된 상태 trajectory
%   u_traj (m x T)        - 최적화된 입력 trajectory
%   cost_trace (vector)   - Iteration별 총 비용 기록
% 

% --- 초기화 ---
x_dim = size(x_traj_init, 1);
u_dim = size(u_traj_init, 1);
T = size(u_traj_init, 2);

x_traj = x_traj_init;
u_traj = u_traj_init;

K_seq = cell(1, T);
k_seq = cell(1, T);

cost_trace = zeros(1, ilqr_opts.max_iter);
lambda = ilqr_opts.lambda;

fprintf('--- iLQR Local Update 시작 ---\n');

% --- 메인 iLQR 루프 ---
for iter = 1:ilqr_opts.max_iter
    
    % 1. 현재 Trajectory의 총 비용 계산
    total_cost = 0;
    for t = 1:T
        total_cost = total_cost + cost_funs.stage_cost(x_traj(:, t), u_traj(:, t), t);
    end
    total_cost = total_cost + cost_funs.terminal_cost(x_traj(:, T+1));
    cost_trace(iter) = total_cost;

    % --- 2. Backward Pass ---
    % V_x, V_xx 초기화 (Value function의 1차, 2차 미분)
    V_x = cost_funs.l_x_final(x_traj(:, T+1));         % V_x_T = l_x_final
    V_xx = cost_funs.l_xx_final(x_traj(:, T+1));       % V_xx_T = l_xx_final
    
    back_pass_done = false;
    while ~back_pass_done
        for t = T:-1:1
            xt = x_traj(:, t);
            ut = u_traj(:, t);
            
            % 현재 시점의 비용 미분
            l_x = cost_funs.l_x(xt, ut, t);
            l_u = cost_funs.l_u(xt, ut, t);
            l_xx = cost_funs.l_xx(xt, ut, t);
            l_uu = cost_funs.l_uu(xt, ut, t);
            l_ux = cost_funs.l_ux(xt, ut, t);
            
            % 현재 시점의 Dynamics
            At = A_seq{t};
            Bt = B_seq{t};
            
            % Action-Value Function (Q-function)의 2차 근사 계수 계산
            % Q(dx,du) ≈ const + Q_x'*dx + Q_u'*du + 0.5*[dx;du]''*[Q_xx, Q_xu; Q_ux, Q_uu]*[dx;du]
            Q_x = l_x + At' * V_x;                      % Q_x = l_x + f_x' * V_x''
            Q_u = l_u + Bt' * V_x;                      % Q_u = l_u + f_u' * V_x''
            Q_xx = l_xx + At' * V_xx * At;              % Q_xx = l_xx + f_x' * V_xx * f_x
            Q_uu = l_uu + Bt' * V_xx * Bt;              % Q_uu = l_uu + f_u' * V_xx * f_u
            Q_ux = l_ux + Bt' * V_xx * At;              % Q_ux = l_ux + f_u' * V_xx * f_x
            
            % Q_uu Regularization (Levenberg-Marquardt)
            Q_uu_reg = Q_uu + eye(u_dim) * lambda;
            
            % 컨트롤러 계산
            try
                % Cholesky decomposition으로 positive definite 확인
                [R_chol, p] = chol(Q_uu_reg);
                if p > 0
                    error('Q_uu_reg is not positive definite');
                end
                
                % k = -inv(Q_uu_reg) * Q_u
                % K = -inv(Q_uu_reg) * Q_ux
                k_t = -R_chol \ (R_chol' \ Q_u);
                K_t = -R_chol \ (R_chol' \ Q_ux);
                
            catch
                % Cholesky 분해 실패 시, 다음 백패스에서 lambda를 높여 재시도
                back_pass_done = false;
                break; % 현재 t 루프 중단
            end
            
            k_seq{t} = k_t;
            K_seq{t} = K_t;
            
            % 다음 시점(t-1)을 위한 V_x, V_xx 업데이트
            V_x = Q_x + K_t' * Q_uu * k_t + K_t' * Q_u + Q_ux' * k_t;
            V_xx = Q_xx + K_t' * Q_uu * K_t + K_t' * Q_ux + Q_ux' * K_t;
            
            if t == 1
                back_pass_done = true;
            end
        end
        
        if ~back_pass_done
            % Regularization을 높여서 재시도
            lambda = lambda * ilqr_opts.lambda_factor;
            fprintf('  [INFO] Backward pass failed. Increasing lambda to %.4e\n', lambda);
            if lambda > 1e6
                fprintf('  [ERROR] Lambda too high. Aborting.\n');
                cost_trace = cost_trace(1:iter);
                return;
            end
        end
    end

    % --- 3. Forward Pass with Line Search ---
    alpha = 1.0;
    line_search_done = false;
    x_traj_new = zeros(x_dim, T+1);
    u_traj_new = zeros(u_dim, T);
    
    for k_ls = 1:ilqr_opts.line_search_k
        x_traj_new(:, 1) = x_traj(:, 1);
        
        for t = 1:T
            % 제어 입력 계산: u_new = u_bar + alpha*k + K*(x_new - x_bar)
            delta_x = x_traj_new(:, t) - x_traj(:, t);
            u_traj_new(:, t) = u_traj(:, t) + alpha * k_seq{t} + K_seq{t} * delta_x;
            
            % 다음 상태 계산 (Local Linear Dynamics 사용)
            x_traj_new(:, t+1) = A_seq{t} * x_traj_new(:, t) + B_seq{t} * u_traj_new(:, t) + c_seq{t};
        end
        
        % 새로운 Trajectory의 비용 계산
        new_total_cost = 0;
        for t = 1:T
            new_total_cost = new_total_cost + cost_funs.stage_cost(x_traj_new(:, t), u_traj_new(:, t), t);
        end
        new_total_cost = new_total_cost + cost_funs.terminal_cost(x_traj_new(:, T+1));
        
        % 비용 감소 확인
        if new_total_cost < total_cost
            line_search_done = true;
            break;
        end
        
        % Step size(alpha) 감소
        alpha = alpha * ilqr_opts.line_search_a;
    end
    
    if ~line_search_done
        fprintf('  [WARN] Line search failed. Cost did not improve. Terminating.\n');
        cost_trace = cost_trace(1:iter); % 기록된 부분까지만 반환
        return;
    end
    
    % --- 4. 업데이트 및 다음 Iteration 준비 ---
    cost_change = total_cost - new_total_cost;
    
    fprintf('Iter: %2d, Cost: %.4f, Cost Change: %.4f, Lambda: %.2e, Alpha: %.3f\n', iter, new_total_cost, cost_change, lambda, alpha);
    
    % Trajectory 업데이트
    x_traj = x_traj_new;
    u_traj = u_traj_new;
    
    % Regularization 감소
    lambda = lambda / ilqr_opts.lambda_factor;
    if lambda < 1e-6
        lambda = 1e-6;
    end
    
    % 수렴 조건 확인
    if cost_change < ilqr_opts.cost_converge
        fprintf('--- Cost converged. iLQR finished. ---\n');
        break;
    end
end

cost_trace = cost_trace(1:iter); % 최종 iter까지만 저장

if iter == ilqr_opts.max_iter
    fprintf('--- Max iterations reached. iLQR finished. ---\n');
end

end
