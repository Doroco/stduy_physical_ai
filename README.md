Notion 페이지: [Physical AI 학습 로드맵](https://www.notion.so/Physical-AI-2adf8b6477c281cf8780dbce0777354f)

# Physical AI 학습 로드맵

이 리포지토리는 Physical AI 학습 로드맵에 따라 주차별로 학습한 내용과 관련 코드를 정리하는 공간입니다.

> **전체 로드맵 개요**: 기초 로보틱스 → 강화학습 기반 로봇 제어 → 멀티모달 인지 및 LLM 융합 → 실환경 통합 시스템 설계

---

## Week 1: 로봇 제어 및 강화학습 기반 행동 학습

### 학습 목표
- Low-level control을 학습하는 로봇 에이전트의 기본 구조 이해
- 시뮬레이터 상에서 강화학습(RL)으로 정책을 학습하고, 이를 실세계로 옮기는 흐름(Sim-to-Real) 파악

### 리뷰 논문
- **Levine, S. et al. (2016).** *End-to-End Training of Deep Visuomotor Policies. JMLR.*
- **Zhu, Y. et al. (2020).** *RoboSuite: A Modular Simulation Framework for Robot Learning. arXiv:2009.12293.*
- **Kalashnikov, D. et al. (2018).** *QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation. arXiv:1806.10293.*\n- **Peng, X. et al. (2018).** *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization. ICRA.*

### 코드
- 관련 코드는 `Week1` 폴더에 있으며, 세부 내용은 `Week1/README.md`를 참고하세요.

---

## Week 2: High-level Task Planning (언어·추론 기반 제어)

### 학습 목표
- LLM이 로봇의 고수준 언어 명령을 해석하고 계획을 세우는 과정 이해

### 리뷰 논문
- **Ahn et al., (2022).** *Do As I Can, Not As I Say (SayCan).*\n- **Driess et al., (2023).** *PaLM-E: An Embodied Multimodal Language Model.*\n- **Zeng et al., (2023).** *VoxPoser: Composable 3D Value Maps for Robotic Manipulation.*\n- **Zeng et al., (2023).** *Large Language Models for Robotics: A Survey. arXiv:2311.07226.*

---

## Week 3: LLM + 강화학습 하이브리드 통합 구조

### 학습 목표
- LLM이 계획을 세우고, RL이 세부 동작을 수행하는 결합 구조 설계 능력 확보

### 리뷰 논문
- **Moroncelli et al., (2024).** *Integrating Reinforcement Learning with Foundation Models for Autonomous Robotics. arXiv:2410.16411.*\n- **Yin et al., (2025).** *Grounding Open-Domain Knowledge from LLMs to Real-World RL Tasks: A Survey. IJCAI.*\n- **Zhang et al., (2025).** *The Landscape of Agentic Reinforcement Learning for LLMs.*\n- **Kim et al., (2024).** *A Survey on Integration of Large Language Models with Intelligent Robots. Springer.*\n\n---\n\n## Week 4: Embodied AI 및 Agentic LLM 기반 로봇\n\n### 학습 목표\n- 실세계에서 스스로 학습하고 의사결정하는 자율형 로봇 설계 개념 이해\n\n### 리뷰 논문\n- **Khan et al., (2025).** *Foundation Model Driven Robotics: A Comprehensive Review.*\n- **Liang et al., (2025).** *Large Model Empowered Embodied AI: A Survey on Decision-Making and Embodied Learning.*\n- **Raptis et al., (2025).** *Agentic LLM-based Robotic Systems: Ethics and Real-World Challenges.*\n- **Zhang et al., (2025).** *Embodied Intelligent Industrial Robotics: Concepts and Techniques.*\n\n---\n