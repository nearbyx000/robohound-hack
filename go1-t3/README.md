# Go1 — T3: ходьба с RL-политикой (+ T2 Safety Block)

Решение заочного этапа Hackathon Go1 (ROBOHACK 2.0).

* **T3** — RL-политика, **самостоятельно обученная командой с нуля** (PPO, MuJoCo, CPU), формирует
  команды 12 суставам по заданной скорости `/cmd_vel`; все команды проходят через Safety Block.
* **T2** — Safety Block (`/lowcmd_bare` → проверки → `/lowcmd`).

```
/cmd_vel ──► go1_rl_policy ──/lowcmd_bare──► go1_safety_block ──/lowcmd──► Simulator
/lowstate ──┘ (50 Гц sim time)               (≈250 Гц, проверки T2)
```

## Контракт запуска (T1)

| Команда | Что делает |
|---|---|
| `/entrypoint.sh ping` | печатает один JSON и завершается с кодом 0 |
| `/entrypoint.sh T2` | запускает Safety Block |
| `/entrypoint.sh T3` | запускает RL-политику + Safety Block |

Ответ `ping`:
```json
{"protocol_version": "1.0", "supported_tasks": ["T2", "T3"], "self_trained_rl": true, "preferred_speed_mps": 0.3}
```

Образ: linux/amd64, Ubuntu 22.04, ROS 2 Humble, `rmw_cyclonedds_cpp`, только CPU, пользователь `ros`
(не root), без интернета во время работы (MuJoCo 3.3.7, сообщения `unitree_go`, модель и веса — внутри образа).
Переменные `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`, `CYCLONEDDS_URI` от запускателя сохраняются.

## Сборка

```bash
docker buildx build --platform linux/amd64 -t go1-solution:latest .
docker run --rm go1-solution:latest ping
```

## Структура

| Путь | Назначение |
|---|---|
| `go1_solution/common.py` | общие константы, наблюдение, PD-закон, пределы — **одинаковы в обучении и в ROS-узлах** |
| `go1_solution/safety_core.py` | логика Safety Block (без ROS): NaN/Inf, пределы, ≤4,5° на сообщение, самоколлизии, потеря потока |
| `go1_solution/safety_node.py` | ROS 2-узел Safety Block |
| `go1_solution/policy.py`, `policy_node.py` | инференс политики на NumPy и ROS 2-узел |
| `models/unitree_go1/` | эталонная модель (go1.xml SHA-256 `d5a74667…525c4a`) и `t2_collision_pairs.csv` |
| `policy/policy.npz` | веса итоговой политики |
| `training/` | среда, конфигурация, обучение, экспорт, оценка |
| `training/runs/go1_walk_ppo/` | **журналы обучения**: `config.yaml`, `progress.csv`, TensorBoard `events.*`, `selection.jsonl` |
| `tests/` | тесты Safety Block, локальный судья T2/T3, прогон эпизодов |

## Safety Block (T2)

* сообщение с NaN/±Inf в любом управляющем поле отклоняется целиком; на выходе остаётся последняя принятая безопасная цель;
* целевые углы ограничиваются диапазонами из ТЗ (4.3);
* цель каждого сустава меняется не более чем на 4,5° между соседними безопасными командами (плавный переход к запрошенной цели);
* конечная конфигурация и весь переход проверяются с шагом ≤ 0,1° на самоколлизии по 655 проверяемым парам
  `t2_collision_pairs.csv` (MuJoCo `mj_collision`, порог проникновения 0,5 мм — строже судейского 1 мм);
  опасная команда отклоняется, удерживается последняя безопасная цель;
* kp ∈ [0, 100], kd ∈ [0, 10], |tau| ≤ пределов моментов Go1;
* при отсутствии `/lowcmd_bare` более 0,3 с sim time — удержание последней принятой цели, dq = 0, tau = 0, kp/kd > 0; поток `/lowcmd` продолжается;
* `/lowcmd` публикуется на каждое второе сообщение `/lowstate` (≈250 Гц sim time) + резервный таймер.

## RL-политика (T3) — самостоятельное обучение

Подтверждающие материалы (ТЗ, 5.4):

1. **Веса модели**: `policy/policy.npz` (NumPy), исходный чекпоинт SB3 — `training/runs/go1_walk_ppo/checkpoints/`.
2. **Конфигурация обучения**: `training/config.yaml` (копия использованной — `training/runs/go1_walk_ppo/config.yaml`).
3. **Журналы и результаты**: `training/runs/go1_walk_ppo/progress.csv`, TensorBoard-логи, `selection.jsonl`
   (выбор чекпоинта), `episode_results.jsonl` (10 локальных эпизодов в публичном симуляторе).

Обучение:
* алгоритм PPO (stable-baselines3 2.3.2, PyTorch CPU), сеть актора/критика 256-256-128 ELU, обучение **с нуля**;
* среда `training/go1_env.py` повторяет публичный симулятор: та же модель menagerie, шаг 0,002 с, старт из keyframe `home`,
  моменты `tau = kp(q* − q) − kd·dq` через `qfrc_applied` с пределами моментов, политика 50 Гц, тот же алгоритм
  плавного перехода Safety Block (≤4,5° за 4 мс), те же наблюдения, что есть в `/lowstate` (без линейной скорости корпуса);
* наблюдение: угловая скорость, проекция гравитации, команда, q − q₀, dq, предыдущее действие; история из 3 кадров (135 значений);
* действие: q* = q₀ + 0,25·a, kp = 30, kd = 0,8;
* рандомизация: трение, масса и центр масс корпуса, kp/kd ±10 %, задержка 0–8 мс, толчки, шум датчиков;
* команды: vx ∈ [0,1; 0,6] м/с с акцентом на заявленные 0,3 м/с.

Повторить обучение и экспорт:
```bash
python3 -m venv venv && . venv/bin/activate
pip install "numpy<2" mujoco==3.3.7 torch stable-baselines3==2.3.2 gymnasium==0.29.1 tensorboard pyyaml
python training/train.py --config training/config.yaml
python training/export_policy.py training/runs/go1_walk_ppo/checkpoints/ppo_<N>_steps.zip policy/policy.npz
python training/eval_policy.py policy/policy.npz --episodes 20
```

## Результаты обучения и локальной проверки

* Обучение: 19,5 млн шагов среды (≈4,5 ч на 8-ядерном CPU), средняя награда эпизода 37,1 (см. `progress.csv`).
  Первая попытка с `SubprocVecEnv` (`training/runs/attempt0_*`) остановлена из-за низкой скорости и заменена
  пакетной средой `training/batched_vec_env.py` с той же конфигурацией.
* Выбор чекпоинта (`selection.jsonl`, 20 эпизодов с рандомизацией каждый): все чекпоинты 10–18 млн — 20/20;
  выбран **16 млн** (быстрее всех: 3 м в среднем за 9,75 с, наименьший боковой увод).
* Публичный симулятор `go1-sim:v1.0` + `run_local.py`, 10 эпизодов с пересозданием контейнеров (`episode_results.jsonl`):
  **10/10**, 3 м за 10,4–10,5 с при 0,3 м/с, боковой увод ≤ 6 мм, крен/тангаж ≤ 1,7°, высота корпуса ≥ 0,275 м,
  `/lowcmd` ≈ 201 Гц, максимальный интервал 14 мс, NaN/Inf нет.
* T2 в публичном симуляторе (`tests/t2_demo_judge.py`): выход в пределах, шаг ≤ 4,5°, нет NaN/Inf, самоколлизия не
  выполняется, при потере потока — удержание последней принятой цели; ≥ 204 Гц, максимальный интервал 12 мс.

## Локальная проверка

```bash
git clone https://github.com/Urika000/go1-simulator-public.git   # публичный симулятор и run_local.py
python tests/test_safety.py                                         # правила T2 (логика)
SIMDIR=go1-simulator-public bash tests/run_t2_demo.sh go1-solution:latest      # T2 в симуляторе
SIMDIR=go1-simulator-public bash tests/run_episodes.sh go1-solution:latest 10 0.3  # 10 эпизодов T3
```
Локальный судья (`tests/local_judge.py`, `tests/t2_demo_judge.py`) — наша реализация публичных правил ТЗ;
официальный Judge закрыт и может отличаться.
