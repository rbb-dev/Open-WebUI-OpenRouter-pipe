# Release notes pending

## Behaviour changes

- **An update now retires the old pipe on every Open WebUI worker** - With several workers, updating the pipe (from the Update tab, by the automatic update, or by saving it in Workspace → Functions) replaced it at once only on the worker that made the change. Every other worker kept running the old version, with its background jobs, until a chat or a model list next reached that worker, which on a quiet worker could take hours. The pipe now tells every worker over Open WebUI's Redis, and each worker retires its old copy as soon as no chat is using it; Open WebUI loads the new version there on the next request. Deleting the pipe stops it on every worker the same way. This needs Redis, as Open WebUI's own multi-worker setup does, and starts working with the first update made after this version is installed, because the copies being replaced must already listen for it.
