import logging
import sys
import time
import os

class Logger:
    """A class to make logging easier
    """
    def __init__(self, name, filename=None):
        """Initialise logger

        Args:
            name (str): name of experiment
            filename (str): File to output log to
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s|%(name)s|%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
            )

        if filename:
            # if the default check that it exists
            if not os.path.isfile(filename):  
                with open(filename, "w") as f:
                    pass
            file_handler = logging.FileHandler(filename)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Instead of print(msg) adding this handler automatically prints the message with the correct format
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setFormatter(formatter)
        self.logger.addHandler(stdout_handler)

        # whether to actually log data
        self.active = True

        self.start_time = 0

        # used to energy values for graphing total energy in the system
        self.energies = []

    def log(self, msg):
        """log a given message

        Args:
            msg (str): message to log
        """
        if self.active:
            self.logger.info(msg)

    def start(self):
        """log start of run
        """
        self.start_time = time.time()
        if self.active:
            self.logger.info("Run Started")

    def end(self):
        """log end of ryb
        """
        sim_run_length = time.time() - self.start_time
        if self.active:
            self.logger.info(f"Run Finished in {sim_run_length//60:.0f}mins {sim_run_length % 60:.2f}s")

    def turn_logging_off(self):
        """if this method is run the simulation is not logged
        """
        self.active = False

if __name__ == "__main__":
    log = Logger("TEST","/home/crae/CompSim/project_v2/orbital_motion.log")
    log.log("Hello World")