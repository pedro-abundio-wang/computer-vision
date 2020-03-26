# computer-vision

## Jupyter Setup

**Installing Anaconda:**
If you decide to work locally, we recommend using the free [Anaconda Python distribution](https://www.anaconda.com/download/), which provides an easy way for you to handle package dependencies. Please be sure to download the Python 3 version.

**Anaconda Virtual environment:**
Once you have Anaconda installed, it makes sense to create a virtual environment for the course. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run (in a terminal)

`conda create -n computer-vision python=3.6.9`

to create an environment called `computer-vision`.

Then, to activate and enter the environment, run

`conda activate computer-vision`

`jupyter notebook --ip=127.0.0.1`

To exit, you can simply close the window, or run

`conda deactivate computer-vision`

You may refer to [this page](https://conda.io/docs/user-guide/tasks/manage-environments.html) for more detailed instructions on managing virtual environments with Anaconda.

## Jekyll Setup

`JEKYLL_ENV=production jekyll build`

`JEKYLL_ENV=production jekyll server --host 127.0.0.1 --port 4000 --detach`
