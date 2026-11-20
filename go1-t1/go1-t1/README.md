# Hackathon Go1 — T1

Контейнер Participant для обязательной проверки T1. Реализован режим `ping`.
В комплекте описано задание T1, поэтому массив `supported_tasks` пуст и
`preferred_speed_mps` имеет значение `null`.

## Сборка

Нужны Docker с Buildx, Python 3 для проверок и доступ к Docker Hub и репозиторию
пакетов ROS 2 на этапе сборки. Linux/amd64 задаётся явно, в том числе при сборке
на Mac с Apple Silicon. На ARM-хосте сборщику нужна поддержка эмуляции amd64.
На Windows запускайте команды из WSL с интеграцией Docker Desktop.

```sh
docker buildx build --platform linux/amd64 --load -t go1-participant:t1 .
```

Образ использует Ubuntu 22.04 и ROS 2 Humble. Cyclone DDS устанавливается при
сборке. При запуске контейнер работает с UID/GID 1000 и использует CPU.

## Проверка

```sh
chmod +x entrypoint.sh
python3 verify_t1.py
python3 verify_t1.py go1-participant:t1
```

Первая команда проверяет скрипт на хосте. Вторая сверяет архитектуру, проверяет
оба способа вызова `/entrypoint.sh ping`, ограничивает каждый вызов десятью
секундами и запускает ROS 2-узел с Cyclone DDS без сети. Все контейнерные
проверки выполняются с `--pull=never --network=none`.

Ожидаемый stdout команды `ping`:

```json
{"protocol_version":"1.0","supported_tasks":[],"self_trained_rl":false,"preferred_speed_mps":null}
```

## Публикация и сдача

1. В архиве рядом с каталогом исходников есть `go1-t1.bundle` с готовым
   Git-коммитом. Из каталога распаковки выполните команды. Подставьте URL
   пустого публичного GitHub-репозитория команды.

   ```sh
   git clone ./go1-t1.bundle go1-t1-repo
   cd go1-t1-repo
   git remote set-url origin "$GITHUB_REPOSITORY_URL"
   git push -u origin main
   git rev-parse HEAD
   ```

   Для сдачи укажите ссылку на этот коммит в GitHub.
2. Уточните адрес репозитория образов и способ авторизации по документу
   организаторов «Инструкции по подготовке и сдаче решения». После авторизации
   задайте переменную `REGISTRY_REPOSITORY` полным адресом репозитория образов
   CookCloud Registry без тега. Из чистой копии указанного коммита выполните:

   ```sh
   : "${REGISTRY_REPOSITORY:?Укажите адрес репозитория образов CookCloud Registry}"
   commit="$(git rev-parse HEAD)"
   image="${REGISTRY_REPOSITORY}:${commit}"
   docker buildx build --platform linux/amd64 --load -t "$image" . &&
     python3 verify_t1.py "$image" &&
     docker push "$image" &&
     docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$image"
   ```

3. Выберите из вывода адрес своего репозитория в форме
   `<репозиторий-образов>@sha256:<digest>`. Проверьте его загрузку через
   `docker pull --platform linux/amd64 <адрес-по-digest>` и передайте
   организаторам этот адрес вместе со ссылкой на конкретный GitHub-коммит.

[Официальные образы ROS](https://hub.docker.com/_/ros) · [Документация Docker Buildx](https://docs.docker.com/reference/cli/docker/buildx/build/)
