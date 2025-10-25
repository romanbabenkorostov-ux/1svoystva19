# app.py — Умная система моделирования 19 почвенных параметров
import numpy as np
import joblib
import os
import itertools
from tabulate import tabulate

# === ПРАВИЛЬНАЯ проверка Streamlit ===
def is_streamlit():
    try:
        import streamlit as st
        return hasattr(st, 'runtime') and st.runtime.exists()
    except:
        return False

# === Пути ===
if os.path.exists('models/soil_predictor.pkl'):
    model_path = 'models/soil_predictor.pkl'
elif os.path.exists('../models/soil_predictor.pkl'):
    model_path = '../models/soil_predictor.pkl'
else:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(project_root, 'models', 'soil_predictor.pkl')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Модель не найдена: {model_path}")

model = joblib.load(model_path)

# === ВСЕ 19 ПАРАМЕТРОВ (в порядке модели!) ===
PARAMS = [
    'Сорг.%',                     # 0
    'Мин. N',                     # 1
    'pH',                         # 2
    'ППВ %',                      # 3
    'Пористость %',               # 4
    'Водопрочные агрегаты %',     # 5
    'N-NO₃',                      # 6
    'N-NH₄',                      # 7
    'P₂O₅',                       # 8
    'K₂O',                        # 9
    'Физ. глина %',               # 10
    'Ил %',                       # 11
    'Микропоры',                  # 12
    'Мезопоры',                   # 13
    'Макропоры',                  # 14
    'Запасы Сорг',                # 15
    'Ca²⁺',                       # 16
    'Mg²⁺',                       # 17
    'МГВ %'                       # 18
]

# === Факторы (5 бинарных) ===
action_names = ['Растения', 'Загрязнение', 'Биочар', 'Нитрификаторы', 'ПАУ-деструкторы']
combos = list(itertools.product([0, 1], repeat=5))

# === Предсказание ===
def predict(combo):
    X = np.array([combo])
    pred = model.predict(X)[0]
    return {param: round(float(pred[i]), 3) for i, param in enumerate(PARAMS)}

# === Все варианты (32) ===
all_variants = [(combo, predict(combo)) for combo in combos]

# === Умный диапазон ===
def get_range(values):
    if not values:
        return "—"
    vals = sorted(set(values))
    if len(vals) == 1:
        return f"{vals[0]:.3f}".rstrip('0').rstrip('.')
    
    intervals = []
    start = prev = vals[0]
    for v in vals[1:]:
        if abs(v - prev) > 1e-3:
            intervals.append(format_interval(start, prev))
            start = v
        prev = v
    intervals.append(format_interval(start, prev))
    return "; ".join(intervals)

def format_interval(start, end):
    if abs(start - end) < 1e-3:
        return f"{start:.3f}".rstrip('0').rstrip('.')
    else:
        s = f"{start:.3f}".rstrip('0').rstrip('.')
        e = f"{end:.3f}".rstrip('0').rstrip('.')
        return f"{s}–{e}"

# === Фильтрация ===
def filter_variants(variants, param, target, tolerance=0.15):
    return [v for v in variants if abs(v[1][param] - target) <= tolerance * max(1, abs(target))]

# === Топ-3 ===
def get_top3(variants, targets):
    scored = []
    for combo, pred in variants:
        error = sum(abs(pred[p] - targets.get(p, pred[p])) for p in targets)
        scored.append((error, combo, pred))
    return sorted(scored)[:3]

# === КОНСОЛЬНЫЙ РЕЖИМ ===
def console_mode():
    current_variants = all_variants.copy()
    targets = {}
    remaining_params = PARAMS.copy()

    print("СИСТЕМА МОДЕЛИРОВАНИЯ 19 ПОЧВЕННЫХ ПАРАМЕТРОВ")
    print("=" * 85)

    while remaining_params:
        print(f"\nДОСТУПНЫЕ ПАРАМЕТРЫ ({len(current_variants)} вариантов):")
        print("-" * 65)
        for i, p in enumerate(remaining_params, 1):
            vals = [v[1][p] for v in current_variants]
            rng = get_range(vals)
            print(f"[{i:2}] {p:<35} → [{rng}]")

        choice = input("\nНомер параметра (или Enter): ").strip()
        if not choice:
            break

        try:
            idx = int(choice) - 1
            if not (0 <= idx < len(remaining_params)):
                raise ValueError
            param = remaining_params[idx]
        except:
            print("Неверный номер!")
            continue

        vals = [v[1][param] for v in current_variants]
        min_v, max_v = min(vals), max(vals)
        target_input = input(f"Цель {param} [{get_range(vals)}]: ").strip()
        if not target_input:
            remaining_params.pop(idx)
            continue

        try:
            target = float(target_input)
            if target < min_v or target > max_v:
                target = max(min_v, min(target, max_v))
                print(f"Ближайшее: {target:.3f}")
        except:
            print("Неверное число!")
            continue

        targets[param] = target
        print(f"Фильтрация по {param} = {target}...")
        current_variants = filter_variants(current_variants, param, target)
        
        if not current_variants:
            print("НЕВОЗМОЖНО! Нет вариантов.")
            break

        remaining_params.pop(idx)

        top3 = get_top3(current_variants, targets)
        print("\nТОП-3 ВАРИАНТА:")
        table = []
        for _, combo, pred in top3:
            row = [f"{pred[p]:.3f}" for p in targets]
            table.append(row)
        if table:
            print(tabulate(table, headers=list(targets.keys()), tablefmt="grid"))
        for i, (_, combo, pred) in enumerate(top3, 1):
            actions = " | ".join(f"{k}: {'да' if v else 'нет'}" for k, v in zip(action_names, combo))
            print(f"#{i}: {actions}")
        print()

    if current_variants:
        best_combo, best_pred = min(current_variants, key=lambda x: sum(abs(x[1][p] - targets.get(p, x[1][p])) for p in targets))
        print("\nФИНАЛЬНАЯ РЕКОМЕНДАЦИЯ:")
        actions = " | ".join(f"{k}: {'да' if v else 'нет'}" for k, v in zip(action_names, best_combo))
        print(actions)
        print({k: f"{v:.3f}" for k, v in best_pred.items() if k in targets})
    else:
        print("Цель недостижима.")

# === ВЕБ-РЕЖИМ (Streamlit) ===
def web_mode():
    import streamlit as st
    st.set_page_config(page_title="Почвенный ИИ", layout="wide")
    st.title("Моделирование 19 почвенных параметров")

    if 'current_variants' not in st.session_state:
        st.session_state.current_variants = all_variants.copy()
        st.session_state.targets = {}
        st.session_state.step = 0

    current_variants = st.session_state.current_variants
    targets = st.session_state.targets
    remaining_params = [p for p in PARAMS if p not in targets]

    if not remaining_params:
        st.success("ГОТОВО! Топ-3 рекомендации:")
        top3 = get_top3(current_variants, targets)
        for i, (error, combo, pred) in enumerate(top3, 1):
            with st.expander(f"Вариант {i} — Ошибка: {error:.3f}"):
                cols = st.columns(5)
                for j, (name, val) in enumerate(zip(action_names, combo)):
                    cols[j].write(f"**{name}**: {'да' if val else 'нет'}")
                st.json({k: f"{v:.3f}" for k, v in pred.items() if k in targets})
        if st.button("Начать заново"):
            st.session_state.clear()
            st.rerun()
        return

    param = remaining_params[0]
    vals = [v[1][param] for v in current_variants]
    min_v, max_v = min(vals), max(vals)
    rng = get_range(vals)

    st.markdown(f"### Шаг {st.session_state.step + 1}: {param}")
    st.write(f"**Допустимо:** `{rng}`")

    # Адаптивный шаг слайдера
    step = 0.01
    if param in ['pH', 'МГВ %']:
        step = 0.01
    elif param in ['Мин. N', 'N-NO₃', 'N-NH₄', 'P₂O₅', 'K₂O', 'Ca²⁺', 'Mg²⁺']:
        step = 0.5
    elif param in ['Сорг.%', 'ППВ %', 'Пористость %', 'Водопрочные агрегаты %', 'Физ. глина %', 'Ил %']:
        step = 0.1
    elif 'поры' in param.lower():
        step = 0.1

    target = st.slider(
        "Выберите цель",
        min_v, max_v, (min_v + max_v) / 2, step=step,
        key=f"slider_{param}_{st.session_state.step}"
    )

    col1, col2 = st.columns(2)
    if col1.button("Применить", type="primary"):
        targets[param] = target
        current_variants = filter_variants(current_variants, param, target)
        if not current_variants:
            st.error("Нет вариантов! Попробуйте другое значение.")
        else:
            st.session_state.current_variants = current_variants
            st.session_state.targets = targets
            st.session_state.step += 1
            st.success(f"Осталось: {len(current_variants)} вариантов")
            st.rerun()

    if col2.button("Пропустить"):
        st.session_state.step += 1
        st.rerun()

    st.progress(st.session_state.step / len(PARAMS))

# === ЗАПУСК ===
if __name__ == '__main__':
    if is_streamlit():
        web_mode()
    else:
        console_mode()
