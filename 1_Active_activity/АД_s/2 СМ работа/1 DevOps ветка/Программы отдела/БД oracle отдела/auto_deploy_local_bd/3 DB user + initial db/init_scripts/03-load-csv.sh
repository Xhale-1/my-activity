#!/usr/bin/env bash
set -euo pipefail

# Конфиг
DATA_DIR="/opt/oracle/scripts/custom/data"
LOG_DIR="${DATA_DIR}/logs"
mkdir -p "$LOG_DIR"

# Параметры подключения
: "${ORACLE_PDB:=FREEPDB1}"
DB_CONN="E3_ADMIN/ddbadmine3@//localhost:1521/${ORACLE_PDB}"

# SQL*Loader в UTF-8 и с точкой как десятичным разделителем
export NLS_LANG=.AL32UTF8
export NLS_NUMERIC_CHARACTERS='. '

# Режим загрузки (TRUNCATE или APPEND)
LOAD_MODE="${DATA_LOAD_MODE:-TRUNCATE}"

# Проверим, что sqlldr доступен
if ! command -v sqlldr >/dev/null 2>&1; then
  echo "sqlldr (SQL*Loader) not found in PATH. Make sure it's installed in the Oracle image."
  exit 1
fi

# Функция: получить реальное имя таблицы (с учётом регистра и кавычек)
get_real_table_name() {
  local t="$1"
  sqlplus -L -s "$DB_CONN" <<SQL | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
set pages 0 feed off head off verify off
select table_name
from user_tables
where table_name = '$t' or table_name = upper('$t');
SQL
}


# Функция: построить спецификацию колонок для CTL (исправленная версия 2.0)
build_colspec() {
  local tname="$1"
  sqlplus -L -s "$DB_CONN" <<SQL
set pages 0 feed off head off verify off trims on
select
  case
    when data_type in ('VARCHAR2','NVARCHAR2','CHAR','NCHAR','CLOB','NCLOB') then
      '"'||column_name||'" CHAR(4000)'
    
    when data_type in ('NUMBER','FLOAT','BINARY_FLOAT','BINARY_DOUBLE') then
      '"'||column_name||'_IN" FILLER CHAR(4000),' || chr(10) ||
      '"'||column_name||'" EXPRESSION "case when upper(trim(:'||column_name||'_IN)) = ''TRUE'' then 1 when upper(trim(:'||column_name||'_IN)) = ''FALSE'' then 0 when length(trim(:'||column_name||'_IN)) = 0 then null else CAST(:'||column_name||'_IN AS NUMBER) end"'
    
    when data_type = 'DATE' then
      '"'||column_name||'_IN" FILLER CHAR(4000),' || chr(10) ||
      '"'||column_name||'" EXPRESSION "case when length(trim(:'||column_name||'_IN))=0 then null else to_date(:'||column_name||'_IN, ''YYYY-MM-DD HH24:MI:SS'') end"'
      
    when data_type like 'TIMESTAMP%' then
      '"'||column_name||'_IN" FILLER CHAR(4000),' || chr(10) ||
      '"'||column_name||'" EXPRESSION "case when length(trim(:'||column_name||'_IN))=0 then null else to_timestamp(:'||column_name||'_IN, ''YYYY-MM-DD HH24:MI:SS.FF'') end"'
      
    else
      '"'||column_name||'" CHAR(4000)'
  end
from user_tab_columns
where table_name = '$tname'
order by column_id;
SQL
}

# Основной цикл: по всем CSV
shopt -s nullglob nocaseglob
for csv in "${DATA_DIR}"/*.csv; do
  base="$(basename "$csv")"
  table_from_file="${base%.*}"

  real_table="$(get_real_table_name "$table_from_file" || true)"
  if [[ -z "$real_table" ]]; then
    echo "[-] Skip: table for file '$base' not found in E3_ADMIN."
    continue
  fi

  echo "[*] Loading $base -> E3_ADMIN.\"$real_table\" ($LOAD_MODE)"

  colspec="$(build_colspec "$real_table")"
  if [[ -z "$colspec" ]]; then
    echo "   Could not build column spec for table $real_table. Skipping."
    continue
  fi

  ctl="${LOG_DIR}/${table_from_file}.ctl"
  log="${LOG_DIR}/${table_from_file}.log"
  bad="${LOG_DIR}/${table_from_file}.bad"
  dsc="${LOG_DIR}/${table_from_file}.dsc"

  # Генерим CTL
  cat > "$ctl" <<EOF
OPTIONS (skip=1, errors=100000, direct=true, rows=5000, bindsize=10485760, readsize=10485760)
LOAD DATA
CHARACTERSET AL32UTF8
INFILE '$csv'
BADFILE '$bad'
DISCARDFILE '$dsc'
$LOAD_MODE
INTO TABLE E3_ADMIN."$real_table"
-- === ИЗМЕНЕНИЕ ЗДЕСЬ ===
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
$(echo "$colspec" | grep . | sed 's/$/,/' | sed '$ s/,$//')
)
EOF

  # Стартуем загрузку
  sqlldr "$DB_CONN" control="$ctl" log="$log" silent=header,feedback

  # Короткий отчёт
  if [[ -s "$bad" ]]; then
    echo "   WARN: rows rejected (see $bad)."
  fi
  echo "   Log: $log"
done

echo "CSV load finished."