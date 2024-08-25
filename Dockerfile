FROM python:3.11-slim-buster

WORKDIR /app

COPY . /app

# ENV HNSWLIB_NO_NATIVE=1 

RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
 PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

CMD ["gradio", "app.py"]