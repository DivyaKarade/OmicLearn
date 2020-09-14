FROM selenium/standalone-chrome

COPY . .
RUN sudo chmod +x ./tests/tests_installation.sh
RUN ./tests/tests_installation.sh
CMD python3 ./tests/tests.py