# GOV.UK Chat Evaluation

## Config

This directory is for storing config files used for evaluations. 

The ones in the default directory are committed to the repository whereas other files within it, or within other directories, will only be available on your local system.

Make changes to the default files if you want to affect how someone typically runs an evaluation or for a common evaluation scenario. For your own specific testing, use other locations.

A couple of usage tips:

- simple options in config files can typically be overridden with command line arguments when running a task, often reducing the need to create new config files (for example, a task with `generate: true` can be overridden with a `--no-generate` CLI option) - run `--help` against the task for options
- whenever an evaluation is run a config file for that generation is stored in the results directory and can be used again to repeat the same configuration.

