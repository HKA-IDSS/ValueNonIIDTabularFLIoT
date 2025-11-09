# Flower_Experimentation_Framework

Experimentation setup for the research project *Federated Machine Learning Data Governance*. The setup works
by encapsulating the [Flower Federated Learning framework](https://flower.ai/), and extending it with functionality
for simulation of different scenarios and data quality measurements.

## Requirements

In order to use the framework, the following tools are needed:
- [anaconda/miniconda](https://www.anaconda.com/) (python can also be used)
- docker (support for docker-compose needed)

## Installation

Installation instructions are followed assuming you have installed anaconda.

1. (Optional, but preferable) Generate a new environment for anaconda, with Python 3.10 as the version. 

`conda create -n 'name of the environment' --python=3.10`

2. Start the environment. Do this everytime you want to work with the framework.

`conda activate 'name of the environment'`

3. Install all needed packages

> I believe this is the last version. If you ran into problem during the installation, 
> try to remove the version of the packages.

`pip install -r requirements_windows_definitive.txt`

4. Then, start the docker compose file for [Optuna](https://optuna.org/). This is needed, as the first time every configuration is run, 
it requires to run the optimization process.

`docker-compose up --build -d`

## Run Experiment

1. (Optional) Create a new experiment configuration. The possible parameters of the 
experiment configuration can be found in the file [Complete Experiment](https://github.com/JsAntoPe/Flower_Experimentation_Framework/blob/main/experiment_configuration/CompleteExperiment.yaml)
2. Run "python Main.py experiment_config_name"
    - Example: `python Main.py ExperimentFedAvgAdultMLP1`

## Documentation

Please, see the first version of the developers documentation [here](https://github.com/JsAntoPe/Flower_Experimentation_Framework/blob/main/documentation/aggregation_methods/how_to_extend_the_framework.md).

[comment]: <> (## Add your files)

[comment]: <> (- [ ] [Create]&#40;https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file&#41;)

[comment]: <> (  or [upload]&#40;https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file&#41; files)

[comment]: <> (- [ ] [Add files using the command line]&#40;https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line&#41;)

[comment]: <> (  or push an existing Git repository with the following command:)

[comment]: <> (```)

[comment]: <> (cd existing_repo)

[comment]: <> (git remote add origin https://iz-gitlab-01.hs-karlsruhe.de/pejo0001/flower_experimentation_framework.git)

[comment]: <> (git branch -M main)

[comment]: <> (git push -uf origin main)

[comment]: <> (```)

[comment]: <> (## Integrate with your tools)

[comment]: <> (- [ ] [Set up project integrations]&#40;https://iz-gitlab-01.hs-karlsruhe.de/pejo0001/flower_experimentation_framework/-/settings/integrations&#41;)

[comment]: <> (## Collaborate with your team)

[comment]: <> (- [ ] [Invite team members and collaborators]&#40;https://docs.gitlab.com/ee/user/project/members/&#41;)

[comment]: <> (- [ ] [Create a new merge request]&#40;https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html&#41;)

[comment]: <> (- [ ] [Automatically close issues from merge requests]&#40;https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically&#41;)

[comment]: <> (- [ ] [Enable merge request approvals]&#40;https://docs.gitlab.com/ee/user/project/merge_requests/approvals/&#41;)

[comment]: <> (- [ ] [Automatically merge when pipeline succeeds]&#40;https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html&#41;)

[comment]: <> (## Test and Deploy)

[comment]: <> (Use the built-in continuous integration in GitLab.)

[comment]: <> (- [ ] [Get started with GitLab CI/CD]&#40;https://docs.gitlab.com/ee/ci/quick_start/index.html&#41;)

[comment]: <> (- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing&#40;SAST&#41;]&#40;https://docs.gitlab.com/ee/user/application_security/sast/&#41;)

[comment]: <> (- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy]&#40;https://docs.gitlab.com/ee/topics/autodevops/requirements.html&#41;)

[comment]: <> (- [ ] [Use pull-based deployments for improved Kubernetes management]&#40;https://docs.gitlab.com/ee/user/clusters/agent/&#41;)

[comment]: <> (- [ ] [Set up protected environments]&#40;https://docs.gitlab.com/ee/ci/environments/protected_environments.html&#41;)

[comment]: <> (***)

[comment]: <> (# Editing this README)

[comment]: <> (When you're ready to make this README your own, just edit this file and use the handy template below &#40;or feel free to)

[comment]: <> (structure it however you want - this is just a starting point!&#41;. Thank you)

[comment]: <> (to [makeareadme.com]&#40;https://www.makeareadme.com/&#41; for this template.)

[comment]: <> (## Suggestions for a good README)

[comment]: <> (Every project is different, so consider which of these sections apply to yours. The sections used in the template are)

[comment]: <> (suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long)

[comment]: <> (is better than too short. If you think your README is too long, consider utilizing another form of documentation rather)

[comment]: <> (than cutting out information.)

[comment]: <> (## Name)

[comment]: <> (Choose a self-explaining name for your project.)

[comment]: <> (## Description)

[comment]: <> (Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be)

[comment]: <> (unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your)

[comment]: <> (project, this is a good place to list differentiating factors.)

[comment]: <> (## Badges)

[comment]: <> (On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the)

[comment]: <> (project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.)

[comment]: <> (## Visuals)

[comment]: <> (Depending on what you are making, it can be a good idea to include screenshots or even a video &#40;you'll frequently see)

[comment]: <> (GIFs rather than actual videos&#41;. Tools like ttygif can help, but check out Asciinema for a more sophisticated method.)

[comment]: <> (## Installation)

[comment]: <> (Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew.)

[comment]: <> (However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing)

[comment]: <> (specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a)

[comment]: <> (specific context like a particular programming language version or operating system or has dependencies that have to be)

[comment]: <> (installed manually, also add a Requirements subsection.)

[comment]: <> (## Usage)

[comment]: <> (Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of)

[comment]: <> (usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably)

[comment]: <> (include in the README.)

[comment]: <> (## Support)

[comment]: <> (Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address,)

[comment]: <> (etc.)

[comment]: <> (## Roadmap)

[comment]: <> (If you have ideas for releases in the future, it is a good idea to list them in the README.)

[comment]: <> (## Contributing)

[comment]: <> (State if you are open to contributions and what your requirements are for accepting them.)

[comment]: <> (For people who want to make changes to your project, it's helpful to have some documentation on how to get started.)

[comment]: <> (Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps)

[comment]: <> (explicit. These instructions could also be useful to your future self.)

[comment]: <> (You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce)

[comment]: <> (the likelihood that the changes inadvertently break something. Having instructions for running tests is especially)

[comment]: <> (helpful if it requires external setup, such as starting a Selenium server for testing in a browser.)

[comment]: <> (## Authors and acknowledgment)

[comment]: <> (Show your appreciation to those who have contributed to the project.)

[comment]: <> (## License)

[comment]: <> (For open source projects, say how it is licensed.)

[comment]: <> (## Project status)

[comment]: <> (If you have run out of energy or time for your project, put a note at the top of the README saying that development has)

[comment]: <> (slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or)

[comment]: <> (owner, allowing your project to keep going. You can also make an explicit request for maintainers.)
