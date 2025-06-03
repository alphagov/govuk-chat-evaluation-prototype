# GOV.UK Chat Evaluation

## Config

This directory is for storing config files used for evaluations. 

They can be specified by providing a `config_path` argument to a task, for example: `uv run govuk_chat_evaluation question_router --config_path config/my_custom_config.yaml`. Each task has a default config that is applied automatically if you don't specify one.

The ones in the default directory are committed to the repository whereas other files within it, or within other directories, will only be available on your local system.

Make changes to the default files if you want to affect how someone typically runs an evaluation or for a common evaluation scenario. For your own specific testing, use this config directory.

A couple of usage tips:

- simple options in config files can typically be overridden with command line arguments when running a task, often reducing the need to create new config files (for example, a task with `generate: true` can be overridden with a `--no-generate` CLI option) - run `--help` against the task for options
- whenever an evaluation is run a config file for that generation is stored in the results directory and can be used again to repeat the same configuration.

