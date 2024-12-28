import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import aiohttp
import asyncio
import requests
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor

WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"
st.set_page_config(page_title="Анализ и мониторинг погоды", layout="wide")


@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


async def get_current_season():
    month = datetime.now().month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"


# синхронный вызов
def get_weather_sync(city, api_key):
    params = {"q": city,
              "appid": api_key,
              "units": "metric"}
    response = requests.get(WEATHER_API_URL, params=params)
    data = response.json()
    if response.status_code != 200:
        st.error(data)
        return None
    data = response.json()
    return data["main"]["temp"]


# асинхронный вызов
async def get_weather_async(city, api_key):
    params = {"q": city,
              "appid": api_key,
              "units": "metric"}
    async with aiohttp.ClientSession() as session:
        async with session.get(WEATHER_API_URL, params=params) as response:
            data = await response.json()
            if response.status != 200:
                st.error(data)
                return None
            return data["main"]["temp"]

async def temp_is_normal(temp, city_stats):
    season = await get_current_season()
    stats = city_stats[city_stats["season"] == season].iloc[0]
    mean, std = stats["mean"], stats["std"]
    is_normal = (mean - 2 * std) <= temp <= (mean + 2 * std)
    return is_normal, mean, std

#Функция написал, а использовать не получилось(
async def check_temperatures_parallel(temps, city_stats):
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(temp_is_normal, temp, city_stats) for temp in temps]
        for future in futures:
            results.append(future.result())
    return results

async def main():
    st.title("Анализ и мониторинг погоды")
    with st.sidebar:
        uploaded_file = st.file_uploader("Загрузить исторические данные", type=["csv"])
        api_key = st.text_input("Введите OpenWeatherMap API Key", type="password")
        use_async = st.checkbox("Использовать асинхронный вызов API", value=True)
    # Если файл не загружен ,то страница пуста
    if uploaded_file:
        df = load_data(uploaded_file)
        cities = sorted(df["city"].unique())
        selected_city = st.selectbox("Выберите город", cities)
        city_data = df[df["city"] == selected_city].copy()
        city_stats = (
            city_data.groupby("season")["temperature"]
            .agg(["mean", "std"])
            .reset_index()
        )
        col1, col2 = st.columns(2)
        with (col1):
            st.subheader("Анализ исторической температуры")
            season_means = city_data.groupby("season")["temperature"].mean()
            season_stds = city_data.groupby("season")["temperature"].std()

            city_data.loc[:, 'season_mean'] = city_data['season'].map(season_means)
            city_data.loc[:, 'season_std'] = city_data['season'].map(season_stds)

            city_data.loc[:, 'anomaly'] = (
                    (city_data['temperature'] < (city_data['season_mean'] - 2 * city_data['season_std'])) |
                    (city_data['temperature'] > (city_data['season_mean'] + 2 * city_data['season_std']))
            )
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=city_data[~city_data['anomaly']]['timestamp'],
                    y=city_data[~city_data['anomaly']]['temperature'],
                    mode="lines",
                    name="Температура",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=city_data[city_data['anomaly']]['timestamp'],
                    y=city_data[city_data['anomaly']]['temperature'],
                    mode="markers",
                    name="Аномалии",
                    marker={"color": "red", "size": 8},
                )
            )

            fig.update_layout(
                title=f"Температура {selected_city}",
                xaxis_title="Дата",
                yaxis_title="Температура (°C)",
            )
            st.plotly_chart(fig)
            st.subheader("Текущая погода")
            if api_key:
                try:
                    start_time = time.time()
                    #тест асинхронного вызова
                    if use_async:
                        current_temperature = await get_weather_async(selected_city, api_key)
                    else:
                        current_temperature = get_weather_sync(selected_city, api_key)
                    if current_temperature:
                        execution_time = time.time() - start_time
                        st.write(f"Текущая температура в {selected_city}: {current_temperature:.1f}°C")
                        st.write(f"Время вызова API: {execution_time:.3f} секунд")
                        is_normal, mean, std = await temp_is_normal(current_temperature, city_stats)
                        status = "нормальная" if is_normal else "аномальная"
                        st.write(f"Текущая температура в данном времене года **{status}**")
                        st.write(f"Средняя температура во времене года: {mean:.1f}°C ± {2 * std:.1f}°C")
                except Exception as e:
                    st.error(f"Ошибка получения температуры: {str(e)}")
            else:
                st.info("Для получения текущей температуры введите API ключ")

        with col2:
            st.subheader("Стастика по временам года")
            season_colors = {"winter": "blue", "spring": "green", "summer": "orange", "fall": "brown"}
            fig = px.box(
                city_data,
                x="season",
                y="temperature",
                color="season",
                color_discrete_map=season_colors,
                title=f"Температура по временам года в {selected_city}"
            )
            fig.update_layout(
                xaxis_title="Времена года",
                yaxis_title="Температура (°C)",
            )
            st.plotly_chart(fig)
            st.write("Статистические показатели:")
            st.dataframe(city_stats)
    else:
        st.write("Загрузите данные в формате CSV! Пример данных:")
        st.write(pd.DataFrame({"timestamp": ["2024-01-01 00:00:00"], "city": ["Sample City"], "temperature": [20.5]}))

if __name__ == "__main__":
    asyncio.run(main())
