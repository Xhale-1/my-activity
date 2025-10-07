import os
import oracledb
from pathlib import Path

# --- НАСТРОЙКИ ---
DB_USER = "E3_ADMIN"
DB_PASS = "ddbadmine3"
DB_DSN = "10.11.77.245:1521/FREEPDB1"

DATA_DIR = r"D:\1a_rudms_work\zadaniya_my\3 ETL DB\python_etl\csv"
# --- КОНЕЦ НАСТРОЕК ---


def map_oracle_type_to_sqlldr(col_info: dict) -> str:
    """Преобразует тип данных Oracle в спецификацию для SQL*Loader."""
    data_type = col_info["DATA_TYPE"].upper()
    char_length = col_info.get("CHAR_LENGTH")
    precision = col_info.get("DATA_PRECISION")
    scale = col_info.get("DATA_SCALE")

    # ---- Обработка бинарных типов ----
    if data_type == "RAW":
        # SQL*Loader понимает тип RAW и автоматически ждет HEX-строку
        return "RAW"
    
    # ---- маппинг обычных типов ----
    if "CHAR" in data_type:  # VARCHAR2, NVARCHAR2, CHAR
        return f'CHAR({char_length})'
    if data_type == "NUMBER":
        if scale is not None and scale > 0:
            return 'DECIMAL EXTERNAL'
        else:  # scale = 0 or None
            return 'INTEGER EXTERNAL'
    if data_type == "DATE":
        return 'DATE "YYYY-MM-DD HH24:MI:SS"'
    if "TIMESTAMP" in data_type:
        return 'TIMESTAMP "YYYY-MM-DD HH24:MI:SS.FF"'
    if data_type == "CLOB":
        return 'CHAR(4000000)'
    
    # Тип по умолчанию на всякий случай
    return 'CHAR(2000)'


def generate_control_files():
    """Генерирует отдельный .ctl.template файл для каждого CSV."""
    
    data_path = Path(DATA_DIR)
    if not data_path.is_dir():
        print(f"Ошибка: Директория '{DATA_DIR}' не найдена!")
        return

    csv_files = sorted([f for f in data_path.iterdir() if f.is_file() and f.suffix.lower() == '.csv'])

    if not csv_files:
        print(f"В директории '{DATA_DIR}' не найдено CSV файлов.")
        return

    connection = None
    try:
        connection = oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN)
        cursor = connection.cursor()
        print("Успешно подключились к базе данных Oracle.")

        generated_count = 0
        for csv_file in csv_files:
            table_name = csv_file.stem
            print(f"Обработка файла: {csv_file.name} для таблицы \"{table_name}\"...")

            sql = """
                SELECT COLUMN_NAME, DATA_TYPE, CHAR_LENGTH, DATA_PRECISION, DATA_SCALE
                FROM ALL_TAB_COLUMNS
                WHERE OWNER = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')
                  AND TABLE_NAME = :table_name
                ORDER BY COLUMN_ID
            """
            cursor.execute(sql, table_name=table_name)
            columns = [dict(zip([d[0] for d in cursor.description], row)) for row in cursor.fetchall()]

            if not columns:
                print(f"  -> Предупреждение: Не найдены столбцы для таблицы \"{table_name}\". Файл пропущен.")
                continue
            
            # --- ИЗМЕНЕННАЯ ЛОГИКА ГЕНЕРАЦИИ СТРОК ---
            column_definitions = []
            for col in columns:
                col_name = col["COLUMN_NAME"]
                data_type = col["DATA_TYPE"].upper()

                if data_type == "BLOB":
                    # Для BLOB-полей генерируем специальную конструкцию из двух строк
                    # 1. Читаем HEX-строку из CSV во временное (FILLER) поле.
                    # 2. Используем SQL-функцию HEXTORAW для преобразования данных из временного поля
                    #    и вставки результата в реальное BLOB-поле.
                    filler_col_name = f"{col_name}_HEX"
                    line1 = f'  {filler_col_name} FILLER CHAR(8000000)'
                    # Стало (правильно):
                    line2 = f'  "{col_name}" EXPRESSION "HEXTORAW(:{filler_col_name})"'
                    column_definitions.append(line1)
                    column_definitions.append(line2)
                else:
                    # Для всех остальных типов данных используем стандартный маппинг
                    loader_type = map_oracle_type_to_sqlldr(col)
                    line = f'  "{col_name}" {loader_type}'
                    column_definitions.append(line)
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---
            
            column_list_str = ",\n".join(column_definitions)

            load_mode = "append"

            ctl_content = f"""options (skip=1)

-- Block for table: {table_name}
LOAD DATA
CHARACTERSET AL32UTF8
infile '/opt/oracle/scripts/custom/data/{csv_file.with_suffix(".csv").name}'
badfile '{csv_file.with_suffix(".bad").name}'
discardfile '{csv_file.with_suffix(".dis").name}'
{load_mode}
into table "{table_name}"
fields terminated by ',' optionally enclosed by '"'
trailing nullcols
(
{column_list_str}
)
"""

            templates_dir = Path("./templates")
            templates_dir.mkdir(parents=True, exist_ok=True)
            ctl_filename = templates_dir / f"{table_name}.ctl.template"
            ctl_filename.write_text(ctl_content, encoding='utf-8')
            print(f"  -> Control-файл '{ctl_filename.name}' успешно создан.")
            generated_count += 1

        print(f"\n✅ Готово! Создано {generated_count} control-файлов в директории: {templates_dir.resolve()}")

    except oracledb.Error as e:
        error, = e.args
        print("❌ Ошибка при работе с Oracle:", error.message)
    finally:
        if connection:
            connection.close()
            print("🔌 Соединение с базой данных закрыто.")


if __name__ == "__main__":
    generate_control_files()