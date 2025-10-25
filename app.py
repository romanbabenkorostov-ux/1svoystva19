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

# === Пути с отладкой ===
if os.path.exists('models/soil_predictor.pkl'):
    model_path = 'models/soil_predictor.pkl'
elif os.path.exists('../models/soil_predictor.pkl'):
    model_path = '../models/soil_predictor.pkl'
else:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(project_root, 'models', 'soil_predictor.pkl')

print(f"[DEBUG] Поиск модели: {model_path}")
print(f"[DEBUG] Текущая директория: {os.getcwd()}")

if not os.path.exists(model_path):
    possible_paths = ['soil_predictor.pkl', 'models/soil_predictor.pkl', '../models/soil_predictor.pkl']
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    else:
        raise FileNotFoundError(f"Модель не найдена! Искали: {possible_paths}")

model = joblib.load(model_path)
print(f"[DEBUG] Модель загружена: {model_path}")

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
print(f"[DEBUG] Создано {len(all_variants)} вариантов")

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
    filtered = [v for v in variants if abs(v[1][param] - target) <= tolerance * max(1, abs(target))]
    print(f"[DEBUG] Фильтрация {param}={target}: {len(variants)} → {len(filtered)} вариантов")
    return filtered

# === Топ-3 ===
def get_top3(variants, targets):
    scored = []
    for combo, pred in variants:
        error = sum(abs(pred[p] - targets.get(p, pred[p])) for p in targets)
        scored.append((error, combo, pred))
    return sorted(scored)[:3]

# === Получить диапазоны параметров ===
def get_param_ranges(variants):
    ranges = {}
    for param in PARAMS:
        vals = [v[1][param] for v in variants]
        if vals:
            ranges[param] = {
                'min': min(vals),
                'max': max(vals),
                'values': sorted(set(vals)),
                'range_str': get_range(vals)
            }
    print(f"[DEBUG] Доступные диапазоны для {len(variants)} вариантов:")
    for p, r in ranges.items():
        print(f"  {p}: {r['range_str']}")
    return ranges

# === Адаптивный шаг для слайдера ===
def get_step(param):
    if param in ['pH', 'МГВ %']:
        return 0.01
    elif param in ['Мин. N', 'N-NO₃', 'N-NH₄', 'P₂O₅', 'K₂O', 'Ca²⁺', 'Mg²⁺', 'Запасы Сорг']:
        return 0.5
    elif param in ['Сорг.%', 'ППВ %', 'Пористость %', 'Водопрочные агрегаты %', 'Физ. глина %', 'Ил %']:
        return 0.1
    elif 'поры' in param.lower():
        return 0.01
    else:
        return 0.1

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
        
        ranges = get_param_ranges(current_variants)
        
        for i, p in enumerate(remaining_params, 1):
            print(f"[{i:2}] {p:<35} → [{ranges[p]['range_str']}]")

        choice = input("\nНомер параметра (или Enter для завершения): ").strip()
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

        target_input = input(f"Цель {param} [{ranges[param]['range_str']}]: ").strip()
        if not target_input:
            remaining_params.pop(idx)
            continue

        try:
            target = float(target_input)
            min_v, max_v = ranges[param]['min'], ranges[param]['max']
            if target < min_v or target > max_v:
                target = max(min_v, min(target, max_v))
                print(f"[INFO] Скорректировано до: {target:.3f}")
        except:
            print("Неверное число!")
            continue

        targets[param] = target
        current_variants = filter_variants(current_variants, param, target)
        
        if not current_variants:
            print("[ERROR] НЕВОЗМОЖНО! Нет вариантов.")
            break

        remaining_params.pop(idx)

        top3 = get_top3(current_variants, targets)
        print("\nТОП-3 ВАРИАНТА:")
        table = []
        for _, combo, pred in top3:
            row = [f"{pred[p]:.3f}".rstrip('0').rstrip('.') for p in targets]
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
        print({k: f"{v:.3f}".rstrip('0').rstrip('.') for k, v in best_pred.items() if k in targets})
    else:
        print("Цель недостижима.")

# === ВЕБ-РЕЖИМ (Streamlit) ===
def web_mode():
    import streamlit as st
    import pandas as pd
    
    st.set_page_config(page_title="Почвенный ИИ", layout="wide")
    st.title("🌱 Моделирование 19 почвенных параметров")
    
    st.markdown("""
    Система интеллектуального подбора агротехнических мероприятий для достижения целевых показателей почвы.
    Анализирует влияние 5 факторов на 19 почвенных параметров.
    """)

    # Инициализация состояния
    if 'current_variants' not in st.session_state:
        with st.spinner('🔄 Загрузка модели и расчёт начальных вариантов...'):
            st.session_state.current_variants = all_variants.copy()
            st.session_state.targets = {}
            st.session_state.selected_param = None
            st.session_state.step = 0
            print(f"[DEBUG WEB] Инициализация: {len(all_variants)} вариантов")

    current_variants = st.session_state.current_variants
    targets = st.session_state.targets
    remaining_params = [p for p in PARAMS if p not in targets]

    print(f"[DEBUG WEB] Шаг {st.session_state.step}: {len(current_variants)} вариантов, осталось: {len(remaining_params)}")

    # Финальные рекомендации
    if not remaining_params or len(current_variants) <= 3:
        st.success("✅ ГОТОВО! Рекомендации построены")
        top3 = get_top3(current_variants, targets)
        
        print("[DEBUG WEB] Финальный топ-3:")
        for i, (score, combo, pred) in enumerate(top3, 1):
            print(f"  #{i}: score={score:.3f}")
        
        # Таблица топ-3
        st.markdown("### 🏆 ТОП-3 ВАРИАНТА")
        
        if targets:
            table_data = []
            for i, (error, combo, pred) in enumerate(top3, 1):
                row = {'#': i}
                for p in targets:
                    row[p] = f"{pred[p]:.3f}".rstrip('0').rstrip('.')
                row['Ошибка'] = f"{error:.3f}"
                table_data.append(row)
            
            df = pd.DataFrame(table_data)
            st.dataframe(df, hide_index=True, width='stretch')
        
        # Детальное описание каждого варианта
        st.markdown("---")
        for i, (error, combo, pred) in enumerate(top3, 1):
            actions = " | ".join(f"**{k}**: {'✅ да' if v else '❌ нет'}" for k, v in zip(action_names, combo))
            st.markdown(f"**#{i}:** {actions}")
        
        st.markdown("---")
        
        # Развернутые карточки
        for i, (error, combo, pred) in enumerate(top3, 1):
            with st.expander(f"🔬 Детальный прогноз варианта {i}"):
                st.markdown("#### Агротехнические мероприятия:")
                cols = st.columns(5)
                for j, (name, val) in enumerate(zip(action_names, combo)):
                    emoji = "✅" if val else "❌"
                    cols[j].markdown(f"**{name}**<br>{emoji}", unsafe_allow_html=True)
                
                st.divider()
                st.markdown("#### Прогнозируемые параметры:")
                
                # Группируем параметры по категориям
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Органика и питание:**")
                    st.metric("Сорг.%", f"{pred['Сорг.%']:.2f}")
                    st.metric("Мин. N", f"{pred['Мин. N']:.1f}")
                    st.metric("N-NO₃", f"{pred['N-NO₃']:.1f}")
                    st.metric("N-NH₄", f"{pred['N-NH₄']:.1f}")
                    st.metric("P₂O₅", f"{pred['P₂O₅']:.1f}")
                    st.metric("K₂O", f"{pred['K₂O']:.1f}")
                
                with col2:
                    st.markdown("**Физические свойства:**")
                    st.metric("pH", f"{pred['pH']:.2f}")
                    st.metric("ППВ %", f"{pred['ППВ %']:.1f}")
                    st.metric("Пористость %", f"{pred['Пористость %']:.1f}")
                    st.metric("Водопрочные агрегаты %", f"{pred['Водопрочные агрегаты %']:.1f}")
                    st.metric("МГВ %", f"{pred['МГВ %']:.2f}")
                
                with col3:
                    st.markdown("**Гранулометрия и поры:**")
                    st.metric("Физ. глина %", f"{pred['Физ. глина %']:.1f}")
                    st.metric("Ил %", f"{pred['Ил %']:.1f}")
                    st.metric("Микропоры", f"{pred['Микропоры']:.3f}")
                    st.metric("Мезопоры", f"{pred['Мезопоры']:.3f}")
                    st.metric("Макропоры", f"{pred['Макропоры']:.3f}")
                    st.metric("Ca²⁺", f"{pred['Ca²⁺']:.1f}")
                    st.metric("Mg²⁺", f"{pred['Mg²⁺']:.1f}")
                    st.metric("Запасы Сорг", f"{pred['Запасы Сорг']:.1f}")
        
        if st.button("🔄 Начать заново", type="primary"):
            st.session_state.clear()
            st.rerun()
        return

    # ШАГ 1: Выбор параметра
    if st.session_state.selected_param is None:
        st.markdown(f"### Шаг {st.session_state.step + 1} из {len(PARAMS)}: Выбор параметра")
        st.info(f"📊 Доступно вариантов: **{len(current_variants)}** | Задано параметров: **{len(targets)}**")
        
        # Показываем информацию о пересчёте
        with st.spinner('🔄 Расчёт доступных диапазонов...'):
            ranges = get_param_ranges(current_variants)
        
        # Группируем параметры по категориям
        st.markdown("#### Выберите параметр для настройки:")
        
        categories = {
            "Органика и азот": ['Сорг.%', 'Мин. N', 'N-NO₃', 'N-NH₄', 'Запасы Сорг'],
            "Макроэлементы": ['P₂O₅', 'K₂O', 'Ca²⁺', 'Mg²⁺'],
            "Физические свойства": ['pH', 'ППВ %', 'Пористость %', 'Водопрочные агрегаты %', 'МГВ %'],
            "Гранулометрия": ['Физ. глина %', 'Ил %'],
            "Поровое пространство": ['Микропоры', 'Мезопоры', 'Макропоры']
        }
        
        for category, params in categories.items():
            available = [p for p in params if p in remaining_params]
            if available:
                with st.expander(f"📁 {category} ({len(available)} параметров)", expanded=True):
                    cols = st.columns(min(3, len(available)))
                    for i, param in enumerate(available):
                        with cols[i % 3]:
                            st.caption(f"**{param}**")
                            st.caption(f"Диапазон: {ranges[param]['range_str']}")
                            if st.button(f"Выбрать", key=f"btn_{param}", width='stretch'):
                                st.session_state.selected_param = param
                                print(f"[DEBUG WEB] Выбран параметр: {param}")
                                st.rerun()
        
        st.progress(st.session_state.step / len(PARAMS))
        return

    # ШАГ 2: Ввод значения выбранного параметра
    param = st.session_state.selected_param
    ranges = get_param_ranges(current_variants)
    param_range = ranges[param]
    
    st.markdown(f"### Шаг {st.session_state.step + 1} из {len(PARAMS)}: {param}")
    st.info(f"📊 **Допустимый диапазон:** `{param_range['range_str']}`")
    
    min_v = param_range['min']
    max_v = param_range['max']
    
    # Защита от min = max
    if abs(max_v - min_v) < 1e-6:
        st.warning(f"⚠️ Доступно только одно значение: **{min_v:.3f}**")
        
        col1, col2 = st.columns(2)
        
        if col1.button("✅ Применить это значение", type="primary", width='stretch'):
            print(f"[DEBUG WEB] Применяем единственное значение {param}={min_v}")
            
            with st.spinner(f'🔄 Пересчёт вариантов для {param} = {min_v}...'):
                targets[param] = min_v
                new_variants = filter_variants(current_variants, param, min_v)
            
            st.session_state.current_variants = new_variants
            st.session_state.targets = targets
            st.session_state.selected_param = None
            st.session_state.step += 1
            st.success(f"✅ Осталось вариантов: {len(new_variants)}")
            st.rerun()
        
        if col2.button("⏭️ Пропустить параметр", width='stretch'):
            st.session_state.selected_param = None
            st.session_state.step += 1
            st.rerun()
        return
    
    step = get_step(param)
    target = st.slider(
        "Выберите целевое значение",
        min_value=float(min_v),
        max_value=float(max_v),
        value=float((min_v + max_v) / 2),
        step=float(step),
        key=f"slider_{param}_{st.session_state.step}"
    )

    col1, col2, col3 = st.columns(3)
    
    if col1.button("✅ Применить", type="primary", width='stretch'):
        print(f"[DEBUG WEB] Применяем {param}={target}")
        
        # Показываем окно обработки
        with st.spinner(f'🔄 Пересчёт вариантов для {param} = {target}...'):
            targets[param] = target
            new_variants = filter_variants(current_variants, param, target)
        
        if not new_variants:
            st.error("⚠️ С таким значением нет вариантов! Попробуйте другое.")
        else:
            st.session_state.current_variants = new_variants
            st.session_state.targets = targets
            st.session_state.selected_param = None
            st.session_state.step += 1
            st.success(f"✅ Осталось вариантов: {len(new_variants)}")
            st.rerun()

    if col2.button("🔙 Другой параметр", width='stretch'):
        st.session_state.selected_param = None
        st.rerun()
    
    if col3.button("⏭️ Пропустить", width='stretch'):
        st.session_state.selected_param = None
        st.session_state.step += 1
        st.rerun()

    # Прогресс
    st.progress(st.session_state.step / len(PARAMS))

# === ЗАПУСК ===
if __name__ == '__main__':
    if is_streamlit():
        web_mode()
    else:
        console_mode()
