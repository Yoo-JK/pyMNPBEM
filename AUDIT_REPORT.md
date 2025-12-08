# pyMNPBEM 포팅 감사 보고서 (Audit Report)

원본 MATLAB: https://github.com/Nikolaos-Matthaiakakis/MNPBEM.git
Python 포트: pyMNPBEM

## 요약 (Executive Summary)

| 항목 | 상태 | 비율 |
|------|------|------|
| 핵심 기능 (Core) | ⚠️ 부분 구현 | ~40% |
| Quasistatic 모드 | ✅ 구현됨 | ~70% |
| Retarded 모드 | ❌ 미구현 | 0% |
| Mesh/Shape 생성 | ⚠️ 부분 구현 | ~60% |
| 데모/예제 | ⚠️ 부분 구현 | ~10% |

---

## 1. BEM 솔버 비교

### MATLAB 원본 (@bem* 클래스)
| 클래스 | 파일 수 | 주요 메서드 |
|--------|---------|-------------|
| @bemstat | 9개 | bemstat, solve, field, potential, mldivide, mtimes 등 |
| @bemret | 8개 | bemret, solve, field, potential, mldivide, mtimes 등 |
| plasmonmode.m | 1개 | 플라즈몬 모드 계산 |

### Python pyMNPBEM (bem/)
| 파일 | 상태 | 비고 |
|------|------|------|
| bem_base.py | ✅ | 추상 기본 클래스 |
| bem_stat.py | ✅ | Quasistatic BEM 솔버 |
| bem_ret.py | ❌ | **미구현** - Retarded BEM 없음 |
| factory.py | ✅ | bemsolver 팩토리 함수 |

### ⚠️ 누락된 기능
- **@bemret**: Retarded BEM 솔버 전체 미구현
- **plasmonmode.m**: 플라즈몬 모드 계산 미구현
- **mldivide/mtimes**: 연산자 오버로딩 일부 미구현

---

## 2. Particles 비교

### MATLAB 원본
| 클래스 | 파일 수 | 주요 기능 |
|--------|---------|-----------|
| @particle | 27개 | border, clean, curvature, curved, edges, flip 등 |
| @comparticle | 11개 | closed, deriv, interp, mask, plot 등 |
| @compstruct | 13개 | fieldnames, getfield, arithmetic ops |
| @compoint | - | 평가 포인트 |
| @compound | - | 복합 입자 기본 클래스 |

### Python pyMNPBEM (particles/)
| 파일 | 상태 | 비고 |
|------|------|------|
| particle.py | ⚠️ | 기본 구현됨, 일부 메서드 누락 |
| comparticle.py | ⚠️ | 기본 구현됨, plot/interp 누락 |
| compound.py | ✅ | 구현됨 |
| compstruct.py | ⚠️ | 기본만 구현, 연산자 오버로딩 미구현 |
| compoint.py | ✅ | 구현됨 |
| point.py | ✅ | 구현됨 |
| layer_structure.py | ✅ | 구현됨 |

### ⚠️ Particle 누락 기능
- **edges()**: 엣지 추출 미구현
- **border()**: 경계 추출 미구현
- **quad()/quadpol()**: 쿼드러처 규칙 미구현
- **totriangles()**: 삼각형 변환 미구현
- **vertcat()**: 입자 연결 미구현
- **plot()/plot2()**: 시각화 함수 미구현
- **index34()**: 인덱스 매핑 미구현
- **curved()**: 곡면 보간 부분 구현

---

## 3. Shapes (입자 형상 생성)

### MATLAB 원본 (@particle 메서드 + Misc/+shape)
| 함수 | 설명 |
|------|------|
| trisphere | 구 |
| tricube | 큐브 |
| trirod | 막대 |
| tritorus | 토러스 |
| triellipsoid | 타원체 |
| tricone | 원뿔 |
| trinanodisk | 나노디스크 |
| triplate | 평판 |

### Python pyMNPBEM (particles/shapes/)
| 파일 | 상태 | 비고 |
|------|------|------|
| trisphere.py | ✅ | 구현됨 |
| tricube.py | ✅ | 구현됨 |
| trirod.py | ✅ | 구현됨 |
| tritorus.py | ✅ | 구현됨 |
| trispheresegment.py | ✅ | 구현됨 |
| tripolygon.py | ✅ | 구현됨 |
| utils.py | ✅ | 유틸리티 |

### ❌ 누락된 Shapes
- **triellipsoid**: 타원체 미구현
- **tricone**: 원뿔 미구현
- **trinanodisk**: 나노디스크 미구현
- **triplate**: 평판 미구현
- **tricylinder**: 원통 미구현

---

## 4. Green Function 비교

### MATLAB 원본 (Greenfun/)
| 클래스 | 파일 수 | 용도 |
|--------|---------|------|
| @greenstat | 4개 | Quasistatic Green 함수 |
| @greenret | 5개 | Retarded Green 함수 |
| @compgreenstat | - | 복합 입자용 |
| @compgreenret | - | 복합 입자용 (retarded) |
| +aca/ | - | Adaptive Cross Approximation |
| +coverlayer/ | - | 코팅 레이어 |
| hmatrices/ | - | H-행렬 압축 |
| slicer.m | 1개 | 대규모 행렬 슬라이싱 |

### Python pyMNPBEM (greenfun/)
| 파일 | 상태 | 비고 |
|------|------|------|
| green_stat.py | ✅ | 구현됨 |
| comp_green_stat.py | ✅ | 구현됨 |
| green_ret.py | ❌ | **미구현** |
| comp_green_ret.py | ❌ | **미구현** |

### ❌ 누락된 Green Function 기능
- **@greenret**: Retarded Green 함수 전체 미구현
- **+aca/**: Adaptive Cross Approximation 미구현 (대규모 시뮬레이션 필수)
- **+coverlayer/**: 코팅 레이어 기능 미구현
- **hmatrices/**: H-행렬 압축 미구현 (성능 최적화)
- **slicer.m**: 대규모 행렬 처리 미구현

---

## 5. Material (유전함수)

### MATLAB 원본 (Material/)
| 클래스 | 설명 |
|--------|------|
| @epsconst | 상수 유전함수 |
| @epsdrude | Drude 모델 |
| @epstable | 테이블 보간 |
| epsfun.m | 사용자 정의 |

### Python pyMNPBEM (material/)
| 파일 | 상태 | 비고 |
|------|------|------|
| eps_base.py | ✅ | 기본 클래스 |
| eps_const.py | ✅ | 구현됨 |
| eps_drude.py | ✅ | 구현됨 |
| eps_table.py | ✅ | 구현됨 |
| eps_fun.py | ✅ | 구현됨 |

### ✅ 데이터 파일
| 파일 | MATLAB | Python |
|------|--------|--------|
| gold.dat | ✅ | ✅ |
| silver.dat | ✅ | ✅ |
| goldpalik.dat | ✅ | ✅ |
| silverpalik.dat | ✅ | ✅ |
| copperpalik.dat | ✅ | ✅ |

Material 모듈은 **완전히 구현됨** ✅

---

## 6. Simulation (여기/검출)

### MATLAB 원본 (Simulation/)

#### Static 모드
| 클래스 | 파일 수 | 주요 기능 |
|--------|---------|-----------|
| @planewavestat | 8개 | absorption, extinction, field, scattering 등 |
| @dipolestat | 7개 | decayrate, farfield, field, potential |
| @spectrumstat | 5개 | farfield, scattering |
| @planewavestatlayer | - | 다층 기판 |
| @planewavestatmirror | - | 거울 기판 |
| @dipolestatlayer | - | 다층 쌍극자 |
| @dipolestatmirror | - | 거울 쌍극자 |
| @eelsstat | - | EELS 시뮬레이션 |

#### Retarded 모드
| 클래스 | 설명 |
|--------|------|
| @planewave | 평면파 (retarded) |
| @dipoleret | 쌍극자 (retarded) |
| @spectrumret | 스펙트럼 (retarded) |
| absorption.m, extinction.m, scattering.m | 광학 단면적 |

### Python pyMNPBEM (simulation/)
| 파일 | 상태 | 비고 |
|------|------|------|
| planewave_stat.py | ⚠️ | 기본 구현, farfield 미구현 |
| dipole_stat.py | ⚠️ | 기본 구현, farfield 미구현 |
| spectrum_stat.py | ✅ | 구현됨 |

### ❌ 누락된 Simulation 기능
- **@planewavestatlayer**: 다층 기판 미구현
- **@planewavestatmirror**: 거울 기판 미구현
- **@dipolestatlayer**: 다층 쌍극자 미구현
- **@dipolestatmirror**: 거울 쌍극자 미구현
- **@eelsstat**: EELS (전자빔 에너지 손실) 미구현
- **farfield 계산**: 원거리장 계산 미구현
- **Retarded 모드 전체**: 모든 retarded 시뮬레이션 미구현

---

## 7. Mie Theory

### MATLAB 원본 (Mie/)
| 파일 | 설명 |
|------|------|
| @miestat | Quasistatic Mie (7개 메서드) |
| @mieret | Retarded Mie (7개 메서드) |
| @miegans | Gans 이론 |
| miesolver.m | 팩토리 함수 |
| spharm.m | 구면 조화함수 |
| sphtable.m | 구면 조화 테이블 |
| vecspharm.m | 벡터 구면 조화함수 |

### Python pyMNPBEM (mie/)
| 파일 | 상태 | 비고 |
|------|------|------|
| mie_stat.py | ✅ | 기본 구현됨 |
| factory.py | ✅ | miesolver |
| mie_ret.py | ❌ | **미구현** |

### ❌ 누락된 Mie 기능
- **@mieret**: Retarded Mie 이론 미구현
- **@miegans**: Gans 이론 (타원체) 미구현
- **spharm.m**: 구면 조화함수 미구현
- **sphtable.m**: 구면 조화 테이블 미구현
- **vecspharm.m**: 벡터 구면 조화함수 미구현
- **decayrate()**: 붕괴율 계산 미구현
- **loss()**: 손실 계산 미구현

---

## 8. Misc (유틸리티)

### MATLAB 원본 (Misc/) - 21개+ 파일
| 함수 | Python | 상태 |
|------|--------|------|
| inner.m | helpers.py | ✅ |
| outer.m | helpers.py | ✅ |
| matcross.m | helpers.py | ✅ |
| vecnorm.m | helpers.py | ✅ |
| vecnormalize.m | helpers.py | ✅ |
| matmul.m | helpers.py | ✅ |
| spdiag.m | helpers.py | ✅ |
| bemoptions.m | options.py | ✅ |
| getbemoptions.m | options.py | ✅ |
| units.m | units.py | ✅ |
| arrowplot.m | - | ❌ |
| coneplot.m | - | ❌ |
| coneplot2.m | - | ❌ |
| distmin3.m | - | ❌ |
| getfields.m | - | ❌ |
| multiWaitbar.m | - | ✅ (tqdm) |
| mycolormap.m | - | ❌ |
| nettable.m | - | ❌ |
| particlecursor.m | - | ❌ |
| patchcurvature.m | - | ⚠️ (particle.curvature) |
| subarray.m | - | ❌ |

### ❌ 누락된 유틸리티
- 시각화 함수들 (arrowplot, coneplot, mycolormap 등)
- particlecursor: 입자 인터랙티브 선택
- distmin3: 최소 거리 계산
- nettable: 네트워크 테이블

---

## 9. 완전히 누락된 MATLAB 폴더

### Mesh2d/ - 21개 파일 (전체 미구현)
| 파일 | 설명 |
|------|------|
| mesh2d.m | 2D 메시 생성 |
| meshpoly.m | 다각형 메시 |
| quadtree.m | 쿼드트리 세분화 |
| refine.m | 메시 세분화 |
| smoothmesh.m | 메시 스무딩 |
| quality.m | 메시 품질 평가 |
| ... | 21개 파일 모두 미구현 |

⚠️ **영향**: 복잡한 2D 형상의 3D 메시 생성 불가

### Base/ - 6개 파일 (부분 구현)
| 파일 | Python | 상태 |
|------|--------|------|
| bemsolver.m | bem/factory.py | ✅ |
| planewave.m | simulation/planewave_stat.py | ✅ |
| dipole.m | simulation/dipole_stat.py | ✅ |
| spectrum.m | simulation/spectrum_stat.py | ✅ |
| greenfunction.m | greenfun/ | ⚠️ |
| electronbeam.m | - | ❌ |

### Demo/ - 여러 예제 (대부분 미구현)
Python에서는 `examples/demo_specstat1.py` 1개만 구현됨

### mex/ - C/C++ 최적화 (미구현)
MATLAB의 MEX 파일들은 성능 최적화를 위한 것으로, Python에서는 NumPy/SciPy로 대체됨

---

## 10. 종합 평가

### ✅ 완전히 구현된 부분
1. **Material 모듈**: 모든 유전함수 클래스 구현
2. **기본 Particle**: 기본적인 입자 표현
3. **주요 Shapes**: sphere, cube, rod, torus 등
4. **Quasistatic BEM**: 기본 솔버 동작
5. **Quasistatic Green Function**: 기본 구현
6. **Quasistatic Mie**: 기본 구현
7. **기본 Simulation**: 평면파, 쌍극자, 스펙트럼

### ⚠️ 부분 구현된 부분
1. Particle 클래스 (일부 메서드 누락)
2. CompStruct (연산자 오버로딩 미구현)
3. Simulation (farfield, layer, mirror 미구현)
4. 유틸리티 (시각화 함수 미구현)

### ❌ 미구현 부분 (중요도 높음)
1. **Retarded BEM/Green/Mie** - 대형 입자 시뮬레이션 불가
2. **EELS** - 전자빔 시뮬레이션 불가
3. **Layer/Mirror** - 기판 위 입자 시뮬레이션 불가
4. **Mesh2d** - 복잡한 2D 형상 생성 불가
5. **H-matrices/ACA** - 대규모 시뮬레이션 최적화 불가

---

## 11. 권장 구현 우선순위

### 1단계 (높은 우선순위)
1. ❗ Retarded BEM 솔버 (@bemret)
2. ❗ Retarded Green 함수 (@greenret)
3. ❗ Layer structure 시뮬레이션

### 2단계 (중간 우선순위)
1. Retarded Mie 이론 (@mieret)
2. 추가 입자 형상 (ellipsoid, cone, nanodisk)
3. Farfield 계산 기능
4. EELS 시뮬레이션

### 3단계 (낮은 우선순위)
1. Mesh2d 메시 생성기
2. H-matrices/ACA 최적화
3. 시각화 함수들
4. 추가 데모 예제

---

## 12. 구현 상태 요약

```
MATLAB MNPBEM 기능 대비 Python 구현율

BEM 솔버:       ████████░░░░░░░░░░░░  40%
Particles:      ████████████░░░░░░░░  60%
Shapes:         ████████████░░░░░░░░  60%
Green Function: ████████░░░░░░░░░░░░  40%
Material:       ████████████████████  100%
Simulation:     ████████░░░░░░░░░░░░  40%
Mie Theory:     ██████░░░░░░░░░░░░░░  30%
Utilities:      ████████████░░░░░░░░  60%
Mesh2d:         ░░░░░░░░░░░░░░░░░░░░  0%
Demo/Examples:  ██░░░░░░░░░░░░░░░░░░  10%

총 예상 구현율: ~40-45%
```

**결론**: pyMNPBEM은 MNPBEM의 **quasistatic 모드**에 대한 기본 기능은 구현되어 있지만, **retarded 모드**, **기판 효과**, **EELS** 등 고급 기능들은 대부분 미구현 상태입니다. 소형 나노입자의 기본적인 광학 특성 계산은 가능하지만, 대형 입자나 복잡한 환경에서의 시뮬레이션은 현재 불가능합니다.
