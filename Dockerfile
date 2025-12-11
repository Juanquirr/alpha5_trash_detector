FROM ultralytics/ultralytics:latest

RUN pip install pandas sahi

# RUN addgroup --system --gid 1001 plocania && \
#    adduser --system --uid 1001 --ingroup plocania --home /home/plocania --shell /bin/bash plocania && \
#    chown -R plocania:plocania /home/plocania

RUN mkdir -p /ultralytics/plocania

WORKDIR /ultralytics/plocania

# USER plocania
