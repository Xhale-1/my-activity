#!/bin/bash
set -euo pipefail

echo "--- [1/3] Running initial DB setup as SYSTEM ---"
# Подключаемся к корню (FREE), скрипт сам переключится в FREEPDB1
sqlplus -L -s "system/${ORACLE_PWD}@//localhost:1521/FREE" @/opt/oracle/scripts/custom/01-create-user-and-tablespace.sql
echo "--- DB setup finished successfully ---"
echo ""

echo "--- [2/3] Running application schema script as E3_ADMIN ---"
sqlplus -L -s "E3_ADMIN/ddbadmine3@//localhost:1521/${ORACLE_PDB}" <<EOF
WHENEVER SQLERROR EXIT SQL.SQLCODE;
@/opt/oracle/scripts/custom/E3_ADMIN_scheme2_fixed.sql
EXIT;
EOF







echo "--- [3/3] Loading CSV data into E3_ADMIN ---"
# sqlldr userid=E3_ADMIN/ddbadmine3@localhost:1521/FREEPDB1 control="/opt/oracle/scripts/custom/loader.ctl.template" log="/opt/oracle/scripts/custom/log.log"

# Указываем директорию, где лежат файлы
DATA_DIR="/opt/oracle/scripts/custom/sqlldr_templates"

# Проверяем, существует ли директория
if [ ! -d "$DATA_DIR" ]; then
    echo "Ошибка: Директория $DATA_DIR не найдена."
    exit 1
fi

# Инициализируем пустой массив
tables=()

# Перебираем все файлы *.csv в указанной директории
for filepath in "$DATA_DIR"/*.ctl.template; do
    # Проверяем, нашлись ли файлы, чтобы избежать ошибки, если директория пуста
    [ -e "$filepath" ] || continue

    # Получаем только имя файла из полного пути (например, "Assembly.csv")
    filename=$(basename "$filepath")

    # Удаляем расширение .csv из имени файла (например, "Assembly")
    tablename="${filename%.ctl.template}"

    # Добавляем обработанное имя в массив
    tables+=("$tablename")
done

mkdir -p /opt/oracle/scripts/custom/logs

for tbl in "${tables[@]}"; do
    echo "Loading $tbl..."
    if sqlldr userid=E3_ADMIN/ddbadmine3@localhost:1521/FREEPDB1 \
              control="/opt/oracle/scripts/custom/sqlldr_templates/$tbl.ctl.template" \
              log="/opt/oracle/scripts/custom/logs/$tbl.log"
    then
        echo "✅ $tbl loaded successfully"
    else
        echo "❌ $tbl failed to load (check log), continuing..."
    fi
done

echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>>   DATABASE SETUP IS COMPLETE!      <<<"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"