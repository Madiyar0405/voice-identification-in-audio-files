import asyncio
import logging
import pickle
from io import BytesIO
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sqlalchemy import Column, Integer, String, LargeBinary, create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.future import select

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Настройка базы данных - синхронный engine для создания таблиц
SYNC_SQLALCHEMY_DATABASE_URL = "postgresql://postgres:1234@localhost/ctf"  # Замените на ваши параметры
sync_engine = create_engine(SYNC_SQLALCHEMY_DATABASE_URL, echo=True)
Base = declarative_base()

class VoiceProfile(Base):
    __tablename__ = "voice_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    voice_features = Column(LargeBinary)
    gmm_model = Column(LargeBinary)
    scaler = Column(LargeBinary)

def create_tables_sync():
    try:
        Base.metadata.create_all(bind=sync_engine)
        logger.info("Таблицы успешно созданы")
    except Exception as e:
        logger.error(f"Ошибка при создании таблиц: {e}")

# Создаем и запускаем поток для создания таблиц
threading.Thread(target=create_tables_sync, daemon=True).start()

ASYNC_SQLALCHEMY_DATABASE_URL = "postgresql+asyncpg://postgres:1234@localhost/ctf" 
async_engine = create_async_engine(ASYNC_SQLALCHEMY_DATABASE_URL, echo=True)
async_session_maker = sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)


templates = Jinja2Templates(directory="templates")

# Функции для работы с голосом
def extract_features(audio_data: bytes) -> np.ndarray:
    try:
        y, sr = librosa.load(BytesIO(audio_data), sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc])
        return features.T
    except Exception as e:
        logger.error(f"Ошибка при извлечении признаков: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке аудио: {e}")


async def train_model(features: np.ndarray) -> tuple[GaussianMixture, StandardScaler]:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    gmm = GaussianMixture(n_components=16, covariance_type='diag', random_state=42)
    gmm.fit(scaled_features)
    return gmm, scaler

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=RedirectResponse)
async def post_home(request: Request,
                     user_id: str = Form(...),
                     audio_file: UploadFile = File(...)):
    if user_id == "identify":  # Идентификация
        identified_user = await identify_voice(audio_file)
        return RedirectResponse(f"/result?user={identified_user}", status_code=303) # Редирект на страницу с результатами
    else: # регистрация
        msg = await register_voice(user_id, audio_file)
        return RedirectResponse("/register_success", status_code=303)

@app.get("/register_success", response_class=HTMLResponse)
async def register_success(request: Request):
    return templates.TemplateResponse("register_success.html", {"request": request})

@app.get("/result", response_class=HTMLResponse)
async def show_result(request: Request, user: str = None):
    result = user
    return templates.TemplateResponse("result.html", {"request": request, "result": result})


@app.post("/register_voice/{user_id}")
async def register_voice(user_id: str, audio_file: UploadFile = File(...)):
    async with async_session_maker() as db:
        try:
            content = await audio_file.read()
            features = extract_features(content)
            gmm, scaler = await train_model(features)

            voice_profile = VoiceProfile(
                user_id=user_id,
                voice_features=pickle.dumps(features),
                gmm_model=pickle.dumps(gmm),
                scaler=pickle.dumps(scaler)
            )
            db.add(voice_profile)
            await db.commit()
            return {"message": "Голосовой профиль успешно зарегистрирован"}
        except IntegrityError:
            await db.rollback()
            raise HTTPException(status_code=400, detail="Пользователь с таким ID уже существует")
        except Exception as e:
            await db.rollback()
            logger.error(f"Ошибка при регистрации голоса: {e}")
            raise HTTPException(status_code=500, detail="Ошибка при регистрации голоса")


@app.post("/identify_voice")
async def identify_voice(audio_file: UploadFile = File(...)):
    async with async_session_maker() as db:
        try:
            content = await audio_file.read()
            features = extract_features(content)
            threshold = -1000  # Порог, настройте на своих данных

            profiles = await db.execute(select(VoiceProfile))
            profiles = profiles.scalars().all()


            candidates = []

            for profile in profiles:
                scaler = pickle.loads(profile.scaler)
                scaled_features = scaler.transform(features)
                gmm = pickle.loads(profile.gmm_model)
                score = gmm.score(scaled_features)
                candidates.append({"user_id": profile.user_id, "score": score})

            candidates.sort(key=lambda x: x["score"], reverse=True)

            top_candidates = candidates[:3] #  Top 3 кандидата

            if not top_candidates or top_candidates[0]["score"] < threshold:
                return "unknown"

            return top_candidates

        except Exception as e:
            logger.error(f"Ошибка при идентификации голоса: {e}")
            raise HTTPException(status_code=500, detail="Ошибка при идентификации голоса")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)