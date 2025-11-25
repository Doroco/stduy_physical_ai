function [K_seq, k_seq, x_traj, u_traj, cost_trace, kl_trace] = run_ilqr_with_kl_constraint(x_traj_init, u_traj_init, A_seq, B_seq, c_seq, cost_funs, u_ref_seq, kl_opts)
%run_ilqr_with_kl_constraint - KL 제약을 포함한 iLQR 로컬 컨트롤러 최적화
% 
% KL-Divergence 제약을 u에 대한 2차 패널티로 근사하여 iLQR을 수행합니다.
% Cost: l_kl(x,u) = l(x,u) + 0.5 * lambda_kl * (u - u_ref)'*W*(u - u_ref)
% 
% 입력:
%   x_traj_init, u_traj_init, A_seq, B_seq, c_seq, cost_funs - run_ilqr_local_update와 동일
%   u_ref_seq (m x T)     - KL 제약의 중심이 되는 reference action 시퀀스
%   kl_opts (struct)      - KL 제약 및 iLQR 옵션:
%     .max_iter, .cost_converge, .lambda, .lambda_factor, 
%     .line_search_a, .line_search_k - iLQR 옵션 (run_ilqr_local_update 참고)
%     .lambda_kl      - KL 패널티 가중치 (scalar)
%     .W_kl           - KL 패널티의 2차항 가중치 행렬 (m x m) (e.g., inv(Sigma))
% 
% 출력:
%   K_seq, k_seq, x_traj, u_traj, cost_trace - run_ilqr_local_update와 동일
%   kl_trace (vector)     - Iteration별 총 KL-divergence 근사값 기록
%

% --- 초기화 ---
x_dim = size(x_traj_init, 1);
u_dim = size(u_traj_init, 1);
T = size(u_traj_init, 2);

x_traj = x_traj_init;
u_traj = u_traj_init;

K_seq = cell(1, T);
k_seq = cell(1, T);

cost_trace = zeros(1, kl_opts.max_iter);
kl_trace = zeros(1, kl_opts.max_iter);
lambda = kl_opts.lambda; % Levenberg-Marquardt lambda
lambda_kl = kl_opts.lambda_kl; % KL-penalty lambda

fprintf('--- iLQR with KL-Constraint 시작 (lambda_kl = %.2f) ---\n', lambda_kl);

% --- 메인 iLQR 루프 ---
for iter = 1:kl_opts.max_iter
    
    % 1. 현재 Trajectory의 총 비용 및 KL 계산
    total_cost = 0;
    total_kl = 0;
    for t = 1:T
        ut = u_traj(:, t);
        u_ref = u_ref_seq(:, t);
        
        % KL 패널티를 포함한 비용
        kl_penalty = 0.5 * lambda_kl * (ut - u_ref)' * kl_opts.W_kl * (ut - u_ref);
        total_cost = total_cost + cost_funs.stage_cost(x_traj(:, t), ut, t) + kl_penalty;
        
        % KL 발산 근사값 계산 (실제 리포팅용)
        total_kl = total_kl + 0.5 * (ut - u_ref)' * kl_opts.W_kl * (ut - u_ref);
    end
    total_cost = total_cost + cost_funs.terminal_cost(x_traj(:, T+1));
    cost_trace(iter) = total_cost;
    kl_trace(iter) = total_kl;

    % --- 2. Backward Pass ---
    V_x = cost_funs.l_x_final(x_traj(:, T+1));
    V_xx = cost_funs.l_xx_final(x_traj(:, T+1));
    
    back_pass_done = false;
    while ~back_pass_done
        for t = T:-1:1
            xt = x_traj(:, t);
            ut = u_traj(:, t);
            u_ref = u_ref_seq(:, t);
            
            % 비용 미분 (l_x, l_xx 등)
            l_x = cost_funs.l_x(xt, ut, t);
            l_u = cost_funs.l_u(xt, ut, t);
            l_xx = cost_funs.l_xx(xt, ut, t);
            l_uu = cost_funs.l_uu(xt, ut, t);
            l_ux = cost_funs.l_ux(xt, ut, t);
            
            % *** KL 제약 항을 비용 미분에 추가 ***
            % l_kl = 0.5 * lambda_kl * (u-u_ref)'*W*(u-u_ref)
            % d(l_kl)/du = lambda_kl * W * (u - u_ref)
            % d2(l_kl)/du2 = lambda_kl * W
            l_u = l_u + lambda_kl * kl_opts.W_kl * (ut - u_ref);
            l_uu = l_uu + lambda_kl * kl_opts.W_kl;
            % l_ux는 u_ref가 x에 의존하지 않으면 불변
            
            % Dynamics
            At = A_seq{t};
            Bt = B_seq{t};
            
            % Q-function 2차 근사 계수
            Q_x = l_x + At' * V_x;
            Q_u = l_u + Bt' * V_x;
            Q_xx = l_xx + At' * V_xx * At;
            Q_uu = l_uu + Bt' * V_xx * Bt;
            Q_ux = l_ux + Bt' * V_xx * At;
            
            % Q_uu Regularization
            Q_uu_reg = Q_uu + eye(u_dim) * lambda;
            
            try
                [R_chol, p] = chol(Q_uu_reg);
                if p > 0, error('not positive definite'); end
                k_t = -R_chol \ (R_chol' \ Q_u);
                K_t = -R_chol \ (R_chol' \ Q_ux);
            catch
                back_pass_done = false;
                break;
            end
            
            k_seq{t} = k_t;
            K_seq{t} = K_t;
            
            % V_x, V_xx 업데이트
            V_x = Q_x + K_t' * Q_uu * k_t + K_t' * Q_u + Q_ux' * k_t;
            V_xx = Q_xx + K_t' * Q_uu * K_t + K_t' * Q_ux + Q_ux' * K_t;
            
            if t == 1
                back_pass_done = true;
            end
        end
        
        if ~back_pass_done
            lambda = lambda * kl_opts.lambda_factor;
            fprintf('  [INFO] Backward pass failed. Increasing lambda to %.4e\n', lambda);
            if lambda > 1e6
                fprintf('  [ERROR] Lambda too high. Aborting.\n');
                cost_trace = cost_trace(1:iter);
                kl_trace = kl_trace(1:iter);
                return;
            end
        end
    end

    % --- 3. Forward Pass with Line Search ---
    % (run_ilqr_local_update와 동일한 로직)
    alpha = 1.0;
    line_search_done = false;
    x_traj_new = zeros(x_dim, T+1);
    u_traj_new = zeros(u_dim, T);
    
    for k_ls = 1:kl_opts.line_search_k
        x_traj_new(:, 1) = x_traj(:, 1);
        
        for t = 1:T
            delta_x = x_traj_new(:, t) - x_traj(:, t);
            u_traj_new(:, t) = u_traj(:, t) + alpha * k_seq{t} + K_seq{t} * delta_x;
            x_traj_new(:, t+1) = A_seq{t} * x_traj_new(:, t) + B_seq{t} * u_traj_new(:, t) + c_seq{t};
        end
        
        new_total_cost = 0;
        for t = 1:T
            kl_penalty = 0.5 * lambda_kl * (u_traj_new(:, t) - u_ref_seq(:, t))' * kl_opts.W_kl * (u_traj_new(:, t) - u_ref_seq(:, t));
            new_total_cost = new_total_cost + cost_funs.stage_cost(x_traj_new(:, t), u_traj_new(:, t), t) + kl_penalty;
        end
        new_total_cost = new_total_cost + cost_funs.terminal_cost(x_traj_new(:, T+1));
        
        if new_total_cost < total_cost
            line_search_done = true;
            break;
        end
        alpha = alpha * kl_opts.line_search_a;
    end
    
    if ~line_search_done
        fprintf('  [WARN] Line search failed. Terminating.\n');
        cost_trace = cost_trace(1:iter);
        kl_trace = kl_trace(1:iter);
        return;
    end
    
    % --- 4. 업데이트 및 다음 Iteration 준비 ---
    cost_change = total_cost - new_total_cost;
    
    % 새로운 Trajectory의 KL 계산
    new_total_kl = 0;
    for t = 1:T
        new_total_kl = new_total_kl + 0.5 * (u_traj_new(:, t) - u_ref_seq(:, t))' * kl_opts.W_kl * (u_traj_new(:, t) - u_ref_seq(:, t));
    end
    
    fprintf('Iter: %2d, Cost: %.4f, KL: %.4f, Cost Change: %.4f, Lambda: %.2e, Alpha: %.3f\n', iter, new_total_cost, new_total_kl, cost_change, lambda, alpha);
    
    x_traj = x_traj_new;
    u_traj = u_traj_new;
    
    lambda = lambda / kl_opts.lambda_factor;
    if lambda < 1e-6, lambda = 1e-6; end
    
    if cost_change < kl_opts.cost_converge
        fprintf('--- Cost converged. KL-iLQR finished. ---\n');
        cost_trace(iter+1) = new_total_cost;
        kl_trace(iter+1) = new_total_kl;
        iter = iter + 1;
        break;
    end
end

cost_trace = cost_trace(1:iter);
kl_trace = kl_trace(1:iter);

if iter == kl_opts.max_iter
    fprintf('--- Max iterations reached. KL-iLQR finished. ---\n');
end

end
