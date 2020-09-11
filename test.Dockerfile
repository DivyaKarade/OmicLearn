FROM selenium/standalone-chrome

COPY . .
RUN sudo chmod +x ./tests/installing.sh
RUN ./tests/installing.sh
CMD python3 ./tests/tests.py