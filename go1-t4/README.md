# Т4 — одометрия Unitree Go1

Узел сопоставляет лидарные сканы `/scan`. Ориентация из `/imu/data` задаёт крен, тангаж и курс, а также позволяет выровнять сканы перед сопоставлением. По сканам оценивается перемещение в плоскости пола; высота начала координат условно равна нулю. Решение использует только публичные датчики и не подписывается на `/cmd_vel`.

`/odom` и TF `odom → trunk` публикуются с одной позой и меткой `/clock`, до 50 Гц симуляционного времени. Используются геометрия крепления `base_scan` и совпадающие направления осей `imu_link` и `trunk` из публичной модели Go1.

## Сборка и запуск

Из каталога `go1-t4`:

```sh
docker buildx build --platform linux/amd64 --load -t robprik-go1:t4 .
docker run --rm --network none robprik-go1:t4 ping
```

Образ заявляет только `T4`. Для работы нужен ROS 2 Humble с Cyclone DDS; зависимости уже внутри образа. Узел запускается командой `/entrypoint.sh T4` и использует параметры DDS от запускателя.

Из каталога [публичного симулятора](https://github.com/Urika000/go1-simulator-public):

```sh
python3 run_local.py --task T4 --headless --logs \
  --sim-image registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim:v1.0 \
  --participant-image robprik-go1:t4
```

Запускатель включает Gazebo Highlevel, lockstep и переднюю RGB-D камеру. Команды движения в публичной симуляции подаются отдельно; официальный Judge в неё не входит.

Сценарий `test_gazebo.py` запускается отдельно от Participant. Скопируйте его в контейнер `Simulator`, имя которого печатает `run_local.py`:

```sh
docker cp /path/to/go1-t4/test_gazebo.py SIMULATOR:/tmp/test_gazebo.py
docker exec SIMULATOR bash -c 'source /etc/go1_sim_env.sh; python3 /tmp/test_gazebo.py'
```

Сценарий подаёт команды и сравнивает одометрию с `/odom/ground_truth` публичного симулятора. Он проверяет частоту, совпадение `/odom` и TF, ошибку перемещения и дрейф.

Для проверки алгоритма без ROS установите NumPy и SciPy и запустите `python test_odometry.py`. Тест строит сканы с шумом, пропусками и наклонами корпуса, затем считает ошибку на метровых отрезках и дрейф за 10 секунд. Это синтетическая проверка, а не результат официального Judge.
