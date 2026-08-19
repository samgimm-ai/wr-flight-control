# WaterRocket Flight Control

## 프로젝트 목표
물과 압축공기만을 동력으로 사용하는 물로켓이 60m 거리의 원형 과녁에 수직 착륙하는 가능성 탐색.

## 추진 시스템
- **주 로켓**: 2L+ 물+압축공기 (발사 및 비행)
- **보조 로켓**: 소형 물로켓 × N개 (자세 제어 + 착륙 역추진)

## 폴더 구조
```
sim/         — 물리 시뮬레이션 (추진, 공기역학, 환경)
gnc/         — Guidance, Navigation, Control
viz/         — 시각화 (궤적, 애니메이션)
hardware/    — 라즈베리파이 인터페이스 (미래)
scenarios/   — 시뮬레이션 설정 파일 (YAML)
tests/       — 단위 테스트
data/        — 시뮬레이션 출력 (gitignore)
```

## 개발 순서
1. 물리 시뮬레이션 (현재)
2. 제어 알고리즘 (GNC)
3. 하드웨어 인터페이스
4. 실제 비행 테스트

## 좌표계
- x: 하강 방향 (downrange)
- y: 측면 (lateral)  
- z: 고도 (altitude, 위가 양수)

## 실행
```bash
pip install -r requirements.txt
python main.py
python main.py --scenario scenarios/baseline.yaml
```
