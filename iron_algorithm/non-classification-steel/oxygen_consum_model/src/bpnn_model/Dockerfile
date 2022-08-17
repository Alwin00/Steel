FROM python:3.6
LABEL maintainer=geng20220810
WORKDIR .
COPY * /app/
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
	xlrd==1.2.0 \
	joblib==1.1.0 \
	numpy==1.19.2 \
	pandas==1.1.5 \
	scikit_learn==0.24.2 

VOLUME ['/data_path','/model_path']

CMD ["bash"]