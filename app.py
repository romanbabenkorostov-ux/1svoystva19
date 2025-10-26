# src/app19.py
# Универсальный помощник: консоль + Streamlit
# ИСПРАВЛЕННАЯ ВЕРСИЯ с обработкой ошибок

import streamlit as st
import joblib
import numpy as np
import os
import itertools
from tabulate import tabulate
from typing import List, Tuple, Dict
from datetime import datetime
from io import StringIO
import sys
import time

# === ОПРЕДЕЛЕНИЕ РЕЖИМА (ИСПРАВЛЕНО) ===
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    IN_STREAMLIT = get_script_run_ctx() is not None
except:
    IN_STREAMLIT = False

IS_LOCAL = not IN_STREAMLIT

# === ЛОГИРОВАНИЕ ===
log_buffer = StringIO() if IS_LOCAL else None

def log_print(*args, **kwargs):
    if IS_LOCAL and log_buffer:
        print(*args, **kwargs, file=log_buffer)

def log_input(prompt=""):
    if IS_LOCAL:
        print(prompt, end="")
        if log_buffer:
            log_print(prompt, end="")
        value = input()
        if log_buffer:
            log_print(value)
        return value
    return ""

# === ПАРАМЕТРЫ ===
PARAMS = [
    'Сорг.%', 'Минеральный азот. мг/кг', 'рН', 'ППВ', 'Пористость',
    '>0.25 мм структуры', 'Нитратный азот. мг/кг', 'Аммонийный азот. мг/кг',
    'Подвижный фосфор. мг/кг', 'Подвижный калий. мг/кг', '< 0.01. %',
    '< 0.001 мм. %', 'микропоры остаточные', 'мезопоры влагосохраняющие',
    'макропоры влагопроводящие', 'запасы Сорг', 'Ca2+. ммоль(+)/100г',
    'Mg2+. ммоль(+)/100г', 'МГВ'
]

ACTION_NAMES = ['Растения', 'Загрязнение', 'Биочар', 'Нитрификаторы', 'ПАУ-деструкторы']

# === Поиск модели (С ОБРАБОТКОЙ ОШИБОК) ===
def find_model() -> str:
    candidates = [
        'soil_predictor_19.pkl',  # В корне репозитория
        'models/soil_predictor_19.pkl',
        os.path.join(os.path.dirname(__file__), '..', 'models', 'soil_predictor_19.pkl'),
        '../models/soil_predictor_19.pkl'
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    
    # Если модель не найдена
    error_msg = """
    ⚠️ ОШИБКА: Модель не найдена!
    
    Проверьте:
    1. Файл soil_predictor_19.pkl загружен в репозиторий
    2. Путь к модели правильный
    3. Файл присутствует в папке models/
    
    Искал в: {}
    """.format('\n    '.join(candidates))
    
    raise FileNotFoundError(error_msg)

# === ГЕНЕРАЦИЯ 32 ВАРИАНТОВ (с обработкой ошибок) ===
def get_all_variants():
    try:
        model = joblib.load(find_model())
    except Exception as e:
        if IN_STREAMLIT:
            st.error(f"❌ Ошибка загрузки модели: {e}")
            st.stop()
        else:
            raise
    
    combos = list(itertools.product([0, 1], repeat=5))
    variants = []
    
    if IN_STREAMLIT:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Генерируем варианты...")
        
        for i, combo in enumerate(combos):
            X = np.array([combo])
            pred = model.predict(X)[0]
            pred_dict = {param: round(pred[j], 3) for j, param in enumerate(PARAMS)}
            variants.append((combo, pred_dict))
            progress_bar.progress((i + 1) / len(combos))
            status_text.text(f"Готово: {i + 1}/{len(combos)} вариантов")
        
        progress_bar.empty()
        status_text.empty()
    else:
        for combo in combos:
            X = np.array([combo])
            pred = model.predict(X)[0]
            pred_dict = {param: round(pred[j], 3) for j, param in enumerate(PARAMS)}
            variants.append((combo, pred_dict))
    
    return variants

# === ФИЛЬТРАЦИЯ ===
def filter_variants(variants, param, target, tol=0.2):
    return [v for v in variants if abs(v[1][param] - target) <= tol * max(1, abs(target))]

# === ВЫВОД ТОП-3 С ТАБЛИЦЕЙ СРАВНЕНИЯ ===
def show_top3(variants, targets, is_streamlit=False):
    top3 = sorted(variants, key=lambda x: sum(abs(x[1].get(p, 0) - targets.get(p, x[1].get(p, 0))) for p in targets))[:3]
    for i, (c, p) in enumerate(top3, 1):
        acts = " | ".join(f"{k}: {'да' if v else 'нет'}" for k, v in zip(ACTION_NAMES, c))
        if is_streamlit:
            st.write(f"**Вариант #{i}:** {acts}")
            
            # Создаём таблицу с целевыми значениями
            table_data = []
            for param in PARAMS:
                row = [param, f"{p[param]:.3f}"]
                if param in targets:
                    row.append(f"{targets[param]:.3f}")
                    diff = abs(p[param] - targets[param])
                    row.append(f"{diff:.3f}")
                else:
                    row.append("—")
                    row.append("—")
                table_data.append(row)
            
            import pandas as pd
            df = pd.DataFrame(table_data, columns=["Показатель", "Расчётное", "Целевое", "Отклонение"])
            st.dataframe(df, use_container_width=True)
        else:
            print(f"\n# {i}: {acts}")
            table = []
            for param in PARAMS:
                row = [param, f"{p[param]:.3f}"]
                if param in targets:
                    row.append(f"{targets[param]:.3f}")
                    diff = abs(p[param] - targets[param])
                    row.append(f"{diff:.3f}")
                else:
                    row.append("—")
                    row.append("—")
                table.append(row)
            
            print(tabulate(table, headers=["Показатель", "Расчётное", "Целевое", "Отклонение"], tablefmt="grid"))
            log_print(f"\n# {i}: {acts}")
            log_print(tabulate(table, headers=["Показатель", "Расчётное", "Целевое", "Отклонение"], tablefmt="grid"))

# === ПРОГНОЗ ПО ДЕЙСТВИЯМ ===
def predict_by_actions(actions: Dict[str, int]) -> Dict[str, float]:
    try:
        model = joblib.load(find_model())
    except Exception as e:
        if IN_STREAMLIT:
            st.error(f"❌ Ошибка загрузки модели: {e}")
            st.stop()
        else:
            raise
    
    X = np.array([[actions.get(name, 0) for name in ACTION_NAMES]])
    pred = model.predict(X)[0]
    return {param: round(pred[i], 3) for i, param in enumerate(PARAMS)}

def show_prediction(pred: Dict[str, float], actions: Dict[str, int]):
    table = [[i+1, param, f"{pred[param]:.3f}"] for i, param in enumerate(PARAMS)]
    headers = ["№", "Показатель", "Значение"]
    acts_str = " | ".join(f"{k}: {'да' if v else 'нет'}" for k, v in actions.items())

    if IN_STREAMLIT:
        st.table(table)
        st.success(f"Прогноз для: {acts_str}")
    else:
        print("\n" + "="*60)
        print("ДЕЙСТВИЯ:")
        print(acts_str)
        print("="*60)
        print(tabulate(table, headers, tablefmt="grid"))
        print(f"\nГотово! Прогноз для: {acts_str}")
        log_print("\n" + "="*60)
        log_print("ДЕЙСТВИЯ:")
        log_print(acts_str)
        log_print("="*60)
        log_print(tabulate(table, headers, tablefmt="grid"))
        log_print(f"\nГотово! Прогноз для: {acts_str}")

# === РЕЖИМ 1: ПРОГНОЗ ПО ДЕЙСТВИЯМ ===
def mode_predict():
    if IN_STREAMLIT:
        st.header("🌱 ПРОГНОЗ ПО ДЕЙСТВИЯМ")
    else:
        print("ПРОГНОЗ ПО ДЕЙСТВИЯМ (19 показателей)")
        log_print("ПРОГНОЗ ПО ДЕЙСТВИЯМ (19 показателей)")

    actions = {}
    if IN_STREAMLIT:
        cols = st.columns(5)
        for i, name in enumerate(ACTION_NAMES):
            with cols[i]:
                val = st.checkbox(name, value=False)
                actions[name] = 1 if val else 0
        if st.button("Рассчитать", type="primary"):
            with st.spinner("Предсказание..."):
                pred = predict_by_actions(actions)
                show_prediction(pred, actions)
    else:
        for name in ACTION_NAMES:
            while True:
                val = log_input(f"{name} (да/нет): ").strip().lower()
                if val in ['да', '1', 'yes', 'y']:
                    actions[name] = 1
                    break
                elif val in ['нет', '0', 'no', 'n', '']:
                    actions[name] = 0
                    break
                else:
                    print("Введите: да / нет")
        pred = predict_by_actions(actions)
        show_prediction(pred, actions)

# === РЕЖИМ 2: РЕКОМЕНДАЦИЯ ПО ЦЕЛЯМ ===
def mode_recommend():
    if IN_STREAMLIT:
        st.header("🎯 РЕКОМЕНДАЦИЯ ПО ЦЕЛЯМ")
    else:
        print("РЕКОМЕНДАЦИЯ ПО ЦЕЛЯМ (пошагово)")
        log_print("РЕКОМЕНДАЦИЯ ПО ЦЕЛЯМ (пошагово)")

    # Инициализация
    if IN_STREAMLIT:
        if 'variants' not in st.session_state:
            with st.spinner("Генерация 32 вариантов..."):
                try:
                    st.session_state.variants = get_all_variants()
                    st.session_state.current = st.session_state.variants.copy()
                    st.session_state.targets = {}
                    st.session_state.remaining = PARAMS.copy()
                except Exception as e:
                    st.error(f"❌ Ошибка инициализации: {e}")
                    st.stop()
        
        current = st.session_state.current
        targets = st.session_state.targets
        remaining = st.session_state.remaining
    else:
        if not hasattr(mode_recommend, "initialized"):
            print("Генерация 32 вариантов...")
            mode_recommend.variants = get_all_variants()
            mode_recommend.current = mode_recommend.variants.copy()
            mode_recommend.targets = {}
            mode_recommend.remaining = PARAMS.copy()
            mode_recommend.initialized = True
        current = mode_recommend.current
        targets = mode_recommend.targets
        remaining = mode_recommend.remaining

    # Основной цикл
    while remaining:
        if not IN_STREAMLIT:
            print(f"\nВАРИАНТОВ: {len(current)} | ПАРАМЕТРОВ: {len(remaining)}")
            log_print(f"\nВАРИАНТОВ: {len(current)} | ПАРАМЕТРОВ: {len(remaining)}")
            ranges = {p: (min(v[1][p] for v in current), max(v[1][p] for v in current)) for p in remaining}
            for i, p in enumerate(remaining, 1):
                mn, mx = ranges[p]
                print(f"[{i}] {p} → [{mn:.2f}–{mx:.2f}]")
                log_print(f"[{i}] {p} → [{mn:.2f}–{mx:.2f}]")

            choice = log_input("\nНомер (Enter — конец): ").strip()
            if not choice:
                print("Выход.")
                log_print("Выход.")
                break
            try:
                idx = int(choice) - 1
                param = remaining[idx]
            except:
                print("Ошибка ввода!")
                log_print("Ошибка ввода!")
                continue

            val = log_input(f"Цель {param}: ").strip()
            if not val:
                print("Пропущено.")
                log_print("Пропущено.")
                continue
            try:
                target = float(val)
            except:
                print("Неверное число!")
                log_print("Неверное число!")
                continue

        else:
            st.write(f"**Осталось вариантов:** {len(current)} | **Параметров:** {len(remaining)}")
            ranges = {p: (min(v[1][p] for v in current), max(v[1][p] for v in current)) for p in remaining}
            options = [f"{i+1}. {p} → [{ranges[p][0]:.2f}–{ranges[p][1]:.2f}]" for i, p in enumerate(remaining)]
            
            # Уникальный ключ на основе оставшихся параметров
            select_key = f"param_select_{len(remaining)}_{hash(tuple(remaining))}"
            choice = st.selectbox("Выберите параметр:", options, key=select_key)
            idx = int(choice.split('.')[0]) - 1
            param = remaining[idx]
            
            # Уникальный ключ для каждого параметра
            input_key = f"target_input_{param.replace(' ', '_').replace('.', '_')}_{len(current)}"
            target = st.number_input(f"Цель для **{param}**:", value=ranges[param][0], step=0.1, format="%.3f", key=input_key)
            
            # Две кнопки: Применить и Рассчитать на текущих данных
            col1, col2 = st.columns(2)
            button_key = f"apply_btn_{len(remaining)}_{len(current)}"
            calc_key = f"calc_btn_{len(remaining)}_{len(current)}"
            
            with col1:
                apply_clicked = st.button("➕ Применить и продолжить", type="primary", key=button_key)
            with col2:
                calc_clicked = st.button("✅ Рассчитать на этих данных", type="secondary", key=calc_key, disabled=(len(targets) == 0))
            
            if calc_clicked and targets:
                # Завершаем ввод и показываем результат
                show_top3(current, targets, is_streamlit=True)
                
                # Кнопка сброса
                if st.button("🔄 Сбросить и начать заново", type="secondary"):
                    st.session_state.variants = get_all_variants()
                    st.session_state.current = st.session_state.variants.copy()
                    st.session_state.targets = {}
                    st.session_state.remaining = PARAMS.copy()
                    st.rerun()
                st.stop()
            
            if not apply_clicked:
                st.stop()

        # Применение
        targets[param] = target
        current = filter_variants(current, param, target)
        if not current:
            msg = "⚠️ НЕВОЗМОЖНО! Нет подходящих вариантов."
            if IN_STREAMLIT:
                st.error(msg)
                # Кнопка сброса при ошибке
                if st.button("🔄 Сбросить и начать заново", type="primary"):
                    st.session_state.variants = get_all_variants()
                    st.session_state.current = st.session_state.variants.copy()
                    st.session_state.targets = {}
                    st.session_state.remaining = PARAMS.copy()
                    st.rerun()
                st.stop()
            else:
                print(msg)
                log_print(msg)
            break
        remaining.pop(idx)

        if not IN_STREAMLIT:
            show_top3(current, targets, is_streamlit=False)
        else:
            st.session_state.current = current
            st.session_state.targets = targets
            st.session_state.remaining = remaining
            show_top3(current, targets, is_streamlit=True)
            st.rerun()

    # Финал
    if current and not remaining:
        best = min(current, key=lambda x: sum(abs(x[1].get(p, 0) - targets.get(p, x[1].get(p, 0))) for p in targets))
        acts = " | ".join(f"{k}: {'да' if v else 'нет'}" for k, v in zip(ACTION_NAMES, best[0]))
        
        if IN_STREAMLIT:
            st.success("✅ ФИНАЛЬНАЯ РЕКОМЕНДАЦИЯ:")
            st.write(f"**Действия:** {acts}")
            
            # Итоговая таблица со всеми параметрами
            st.subheader("Итоговая таблица параметров")
            table_data = []
            for param in PARAMS:
                row = [param, f"{best[1][param]:.3f}"]
                if param in targets:
                    row.append(f"{targets[param]:.3f}")
                    diff = abs(best[1][param] - targets[param])
                    row.append(f"{diff:.3f}")
                else:
                    row.append("—")
                    row.append("—")
                table_data.append(row)
            
            import pandas as pd
            df = pd.DataFrame(table_data, columns=["Показатель", "Расчётное", "Целевое", "Отклонение"])
            st.dataframe(df, use_container_width=True)
            
            # Кнопка сброса
            if st.button("🔄 Сбросить и начать заново", type="secondary"):
                st.session_state.variants = get_all_variants()
                st.session_state.current = st.session_state.variants.copy()
                st.session_state.targets = {}
                st.session_state.remaining = PARAMS.copy()
                st.rerun()
        else:
            msg = "\nФИНАЛЬНАЯ РЕКОМЕНДАЦИЯ:\n" + acts
            for k, v in best[1].items():
                msg += f"\n  {k}: {v:.3f}"
            print(msg)
            log_print(msg)

# === КОНСОЛЬНЫЙ РЕЖИМ ===
def run_console():
    print("Моделирование свойств почв по реакции на агротехнические и биотехнологические воздействия")
    print("="*50)
    print("[1] Прогноз по действиям")
    print("[2] Рекомендация по целям")
    choice = log_input("Выберите режим (1 или 2): ").strip()

    if choice == "1":
        mode_predict()
    elif choice == "2":
        mode_recommend()
    else:
        print("Неверный выбор.")
        log_print("Неверный выбор.")

    # === СОХРАНЕНИЕ ЛОГА ===
    if IS_LOCAL and log_buffer:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"app19_log_{timestamp}.txt"
        log_path = os.path.join(os.path.dirname(__file__), log_filename)
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(log_buffer.getvalue())
        print(f"\nЛог сохранён: {log_path}")

# === STREAMLIT РЕЖИМ ===
def run_streamlit():
    st.set_page_config(page_title="Моделирование свойств почв", layout="wide", page_icon="🌱")
    
    st.title("🌱 Моделирование свойств почв по реакции на агротехнические и биотехнологические воздействия")
    
    # Проверка наличия модели
    try:
        find_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("💡 **Решение:** Загрузите файл `soil_predictor_19.pkl` в репозиторий GitHub")
        st.stop()

    mode = st.radio("Режим:", ["Прогноз по действиям", "Рекомендация по целям"], horizontal=True)

    if mode == "Прогноз по действиям":
        mode_predict()
    else:
        mode_recommend()

# === ЗАПУСК ===
if __name__ == "__main__":
    if IN_STREAMLIT:
        run_streamlit()
    else:
        run_console()