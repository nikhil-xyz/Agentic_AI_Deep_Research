FROM python:3.12


RUN useradd -m -u 1000 user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app
ADD --chown=user ./. $HOME/app/.
RUN chown user:user -R $HOME/app

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# RUN python main.py
CMD ["streamlit", "run", "app.py", "–server.port=8501", "–server.address=0.0.0.0"]