FROM ubuntu:18.04
EXPOSE 8503

COPY . .
RUN chmod +x ./tests/installing.sh
RUN ./tests/installing.sh
CMD python3 tests/tests.py