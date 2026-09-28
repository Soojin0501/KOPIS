FROM python:3.12-slim
WORKDIR /work
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
CMD ["python"]
