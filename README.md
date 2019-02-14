# Spider Simulation

### Installation
This project uses `python3`. Preferred way to install this is using a virtual environment (`virtualenv`), so as to not muddle your dependencies.
From inside your virtualenv, run `pip install -r requirements.txt` to install python packages. You will also need `ffmpeg`, which can be installed on Linux with `sudo apt-get install ffmpeg`, or mac with `brew install ffmpeg` on mac. I'm not sure if this will work on Windows.


### Code structure:

* **spider_web.py** This defines the actual web objects, such as nodes, edges, and the web itself.
* **web_zoo.py** This provides utility functions for creating different types of webs, such as circular ones, lines, etc.
* **learninhg_model.py** This contains a TensorFlow model of a spider, that can be trained to better predict outputs.
* **record_data_for_spider_locating.py**
* **spider_data.py** This creates TensorFlow datasets from pickled numpy arrays, for training the spider model.
* **record_web_data** This simulates a web with a spider at the center, and records the sensory data for that spider to a file for later training.
* **record_data_for_spider_locating** This is similar to `record_web_data.py`, except that it records data as if there were a spider at EVERY intersection.
* **spider_guessing_visualization** This is a silly example of moving the spider around on the web and recording it.
* **mpl_visualization.py** This contains the base drawing class for drawing an oscillating web. It is overridden in various places, such as in `spider_locating_visualization.py`
* **spider_locating_visualization.py**  This takes in a trained spider model, and simulates/records the spider finding its prey
* **learn_prey_direction.py** This trains the spider model to determine which direction prey is in, from the center. Uses data generated in `record_web_data.py`
* **learn_prey_location.py** This trains the spider model to determine the LOCATION of the prey. Uses data generated in `record_data_for_spider_locating.py`
