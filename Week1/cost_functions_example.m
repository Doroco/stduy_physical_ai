function cost_funs = cost_functions_example(x_target, Q, R, Q_final)
%cost_functions_example - 예제용 2차 비용 함수 핸들 생성
%
% 이 함수는 iLQR 알고리즘에 필요한 비용 함수와 그 미분(1차, 2차)에 대한
% 함수 핸들을 포함하는 구조체를 생성합니다.
%
% 비용 함수 형태:
% - Stage Cost: l(x,u) = 0.5 * (x - x_target)'*Q*(x - x_target) + 0.5 * u'*R*u
% - Terminal Cost: l_T(x) = 0.5 * (x - x_target)'*Q_final*(x - x_target)
%
% 입력:
%   x_target (n x 1) - 목표 상태 벡터
%   Q (n x n) - 상태에 대한 비용 가중치 행렬 (positive semi-definite)
%   R (m x m) - 입력에 대한 비용 가중치 행렬 (positive definite)
%   Q_final (n x n) - 최종 상태에 대한 비용 가중치 행렬 (positive semi-definite)
%
% 출력:
%   cost_funs (struct) - 다음 필드를 포함하는 함수 핸들 구조체:
%     .stage_cost      - l(x, u, t)
%     .terminal_cost   - l_T(x)
%     .l_x             - stage cost의 x에 대한 1차 미분
%     .l_u             - stage cost의 u에 대한 1차 미분
%     .l_xx            - stage cost의 x에 대한 2차 미분
%     .l_uu            - stage cost의 u에 대한 2차 미분
%     .l_ux            - stage cost의 u, x에 대한 2차 미분
%     .l_x_final       - terminal cost의 x에 대한 1차 미분
%     .l_xx_final      - terminal cost의 x에 대한 2차 미분
%

narginchk(4, 4);

% 함수 핸들 구조체 초기화
cost_funs = struct();

% --- Stage Cost 및 미분 ---
cost_funs.stage_cost   = @(x, u, ~) 0.5 * (x - x_target)' * Q * (x - x_target) + 0.5 * u' * R * u;

% 1차 미분 (Gradients)
cost_funs.l_x  = @(x, ~, ~) Q * (x - x_target); % dl/dx
cost_funs.l_u  = @(~, u, ~) R * u;             % dl/du

% 2차 미분 (Hessians)
cost_funs.l_xx = @(~, ~, ~) Q;                 % d2l/dx2
cost_funs.l_uu = @(~, ~, ~) R;                 % d2l/du2
cost_funs.l_ux = @(~, ~, ~) zeros(size(R, 1), size(Q, 2)); % d2l/dudx

% --- Terminal Cost 및 미분 ---
cost_funs.terminal_cost = @(x) 0.5 * (x - x_target)' * Q_final * (x - x_target);

% 1차 미분 (Gradient)
cost_funs.l_x_final  = @(x) Q_final * (x - x_target); % dl_T/dx

% 2차 미분 (Hessian)
cost_funs.l_xx_final = @(x) Q_final;                  % d2l_T/dx2

end
