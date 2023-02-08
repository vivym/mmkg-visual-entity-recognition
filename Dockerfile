FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /work

RUN pip install --no-cache-dir --upgrade mmkg-visual-entity-recognition

CMD ["uvicorn", "mmkg_visual_entity_recognition:app", "--host", "0.0.0.0", "--port", "80", "--root-path", "/mmkg-visual-entity-recognition"]
