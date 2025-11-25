# Physical AI 학습 로드맵 - 1단계: 강화학습 기반 로봇 제어

이 리포지토리는 "Physical AI 학습 로드맵"의 첫 번째 단계인 **로봇 제어 및 강화학습 기반 행동 학습**에 대한 MATLAB 코드 예제를 포함합니다.

주요 목표는 시뮬레이션 데이터를 통해 로봇의 동역학(dynamics)을 국소적(locally)으로 학습하고, iLQR(Iterative Linear Quadratic Regulator)과 같은 궤적 최적화(trajectory optimization) 알고리즘을 사용하여 제어 정책을 개선하는 과정을 이해하는 것입니다. 이는 Guided Policy Search (GPS)와 같은 고급 강화학습 알고리즘의 핵심 구성 요소입니다.

## 주요 개념

이 코드들은 다음과 같은 핵심 개념을 다룹니다.

- **지역적 동역학 학습 (Learning Local Dynamics):** 복잡한 실제 로봇 동역학을 특정 궤적 주변에서 시변(time-varying) 선형 모델 `x_{t+1} = A_t * x_t + B_t * u_t + c_t`로 근사합니다.
- **iLQR (Iterative Linear Quadratic Regulator):** 학습된 지역적 동역학 모델과 비용 함수를 사용하여, 목표를 더 잘 달성하는 새로운 제어 입력을 반복적으로 찾아 최적의 궤적을 생성합니다.
- **KL-Divergence 제약 (KL-Divergence Constraint):** 정책이 너무 급격하게 변하는 것을 막고 안정적인 학습을 돕기 위해, 새로운 정책이 이전 정책에서 크게 벗어나지 않도록 제한하는 '신뢰 영역(Trust Region)'을 설정합니다.

## 파일 구성 및 설명

이 리포지토리는 `Week1` 폴더에 다음과 같은 주요 MATLAB 스크립트를 포함하고 있습니다.

- **`demo_gps_local_update.m`**: 메인 데모 스크립트입니다. 아래의 모든 과정을 통합하여 Guided Policy Search (GPS)의 지역 정책 업데이트(local policy update) 단계를 시뮬레이션합니다. **이 스크립트를 실행하면 전체 과정을 확인할 수 있습니다.**
- **`learn_local_dynamics_rr.m`**: 가상의 2-DOF 로봇팔(RR arm)을 PD 제어기로 구동하여 궤적 데이터를 수집하고, 특정 시점의 지역적 동역학을 선형 회귀로 추정하는 과정을 보여줍니다.
- **`estimate_local_dynamics.m`**: `learn_local_dynamics_rr.m`에서 생성된 궤적 데이터 전체에 대해 각 시간 스텝별 동역학 행렬(A, B, c)을 추정하는 함수입니다.
- **`run_ilqr_local_update.m`**: 추정된 지역적 동역학을 기반으로 표준 iLQR 알고리즘을 수행하여 최적 제어 정책을 찾는 함수입니다.
- **`run_ilqr_with_kl_constraint.m`**: 표준 iLQR에 KL-Divergence 제약을 추가하여, 정책이 안정적으로 업데이트되도록 개선한 iLQR 알고리즘을 구현한 함수입니다.
- **`cost_functions_example.m`**: iLQR 최적화에 사용될 목표 상태와 비용 함수(State/Terminal Cost)를 정의하는 헬퍼 함수입니다.

## 실행 방법

MATLAB에서 `demo_gps_local_update.m` 스크립트를 실행하면, 데이터 생성, 동역학 추정, iLQR 및 KL-iLQR을 통한 최적화, 결과 비교 시각화까지의 전체 파이프라인을 확인할 수 있습니다.
