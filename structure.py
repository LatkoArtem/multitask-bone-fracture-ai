from pathlib import Path

def print_project_tree(directory=".", ignore_dirs=None, descriptions=None, align_column=40, max_items=10):
    """
    Генерує структуру проєкту у вигляді дерева з коментарями та лімітом файлів у папці.
    """
    if ignore_dirs is None:
        ignore_dirs = {'.git', '__pycache__', 'venv', '.venv', '.env', 'node_modules', '.ipynb_checkpoints', '__init__.py'}
    if descriptions is None:
        descriptions = {}

    base_path = Path(directory)
    print(f"{base_path.name}/")

    def _tree(path, prefix=""):
        # Отримуємо всі файли та папки, фільтруємо ігноровані
        items = list(path.iterdir())
        items = [item for item in items if item.name not in ignore_dirs]
        
        # Сортуємо: спочатку папки, потім файли (за алфавітом)
        items.sort(key=lambda x: (not x.is_dir(), x.name))

        # Перевіряємо, чи перевищує кількість файлів ліміт
        has_more = len(items) > max_items
        display_items = items[:max_items]

        for index, item in enumerate(display_items):
            # Елемент є останнім тільки якщо ми дійшли до кінця списку І немає прихованих файлів
            is_last = (index == len(display_items) - 1) and not has_more
            connector = "└── " if is_last else "├── "
            
            # Формуємо базовий рядок з ім'ям
            item_name = f"{item.name}/" if item.is_dir() else item.name
            line = f"{prefix}{connector}{item_name}"
            
            # Додаємо коментар, якщо він є у словнику
            if item.name in descriptions:
                # Вирівнюємо коментар по заданій колонці
                padding_length = max(align_column - len(line), 2)
                padding = " " * padding_length
                line += f"{padding}# {descriptions[item.name]}"
            
            print(line)
            
            # Якщо це папка, рекурсивно заходимо в неї
            if item.is_dir():
                extension = "    " if is_last else "│   "
                _tree(item, prefix=prefix + extension)

        # Якщо були приховані файли, виводимо три крапки
        if has_more:
            print(f"{prefix}└── ...")

    _tree(base_path)

# --- Приклад використання ---
if __name__ == "__main__":
    my_descriptions = {
        "FracAtlas": "Датасет",
        "data_module.py": "Завантаження та аугментація даних",
        "models.py": "Архітектури нейронних мереж",
        "trainer.py": "Цикли навчання та валідації",
        "inference.py": "Каскадний пайплайн інференсу",
        "utils.py": "Допоміжні функції та метрики",
        "exploration.ipynb": "Розвідковий аналіз даних",
        "config.yaml": "Гіперпараметри",
        "main.py": "Точка входу"
    }

    # max_items=10 гарантує, що виведеться не більше 10 файлів/папок на одному рівні
    print_project_tree(directory=".", descriptions=my_descriptions, max_items=10)