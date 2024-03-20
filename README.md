## Media Forensic Audio Framework Tool

This is an analysis tool for audio data. It will segment the audios based on energy level and then it facilitates features like classification, gender and emotion detection as well as audio authenticity detection. Additionaly, this application will provide an explainatory analysis with visual representaiton, including data types of each step of the process.


## Installation

1. Open a terminal in the project directory, be sure to get inside of the backend directory.

2. Run the following command to set up the project:

```bash
make setup
```

This command will create a Python virtual environment, activate it, and install the dependencies from the `requirements.txt` file.

## Running the Project

Once the setup is complete, you can run the project with the following command:

```bash
make run
```

This command will activate the virtual environment and start the Uvicorn server.

Please note that you need to have `make` installed on your system to use these commands. If you don't have `make` installed, you can install it with your system's package manager. For example, on Ubuntu, you can install `make` with `sudo apt install make`.


This will provide clear instructions for anyone who wants to install and run your project.


## Running Frontend

Go to the web directory and run 

```bash
npm install
```
It will install all the required packages. Futher just run the command

```bash
npm run dev
```
This will run the frontend and it can be accessed in http://localhost:5173
