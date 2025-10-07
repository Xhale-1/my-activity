-- init_db.sql
whenever sqlerror exit failure rollback
set echo on feedback on serveroutput on

alter session set container = FREEPDB1;

-- Создаём табличное пространство, только если его ещё нет
declare
  v_count integer;
begin
  select count(*) into v_count from dba_tablespaces where tablespace_name = 'E3SERIES';
  if v_count = 0 then
    -- === ВОТ ГЛАВНОЕ ИСПРАВЛЕНИЕ ===
    -- Мы явно указываем имя файла. OMF сам подставит правильный путь.
    execute immediate '
      create tablespace E3SERIES
        datafile ''e3series01.dbf'' size 50M
        autoextend on next 10M maxsize 10G';
    dbms_output.put_line('Tablespace E3SERIES created.');
  else
    dbms_output.put_line('Tablespace E3SERIES already exists.');
  end if;
end;
/

-- Создаём пользователя, только если его ещё нет
declare
  v_count integer;
begin
  select count(*) into v_count from dba_users where username = 'E3_ADMIN';
  if v_count = 0 then
    execute immediate 'create user E3_ADMIN identified by "ddbadmine3" default tablespace E3SERIES';
    dbms_output.put_line('User E3_ADMIN created.');
  else
    dbms_output.put_line('User E3_ADMIN already exists.');
  end if;
end;
/

-- Выдаём права
grant connect, resource to E3_ADMIN;
alter user E3_ADMIN quota unlimited on E3SERIES;

-- Проверяем результат
select tablespace_name from dba_tablespaces where tablespace_name='E3SERIES';
select username from dba_users where username='E3_ADMIN';





PROMPT 'Creating application roles...';

declare
  v_count integer;
begin
  -- Создаём роль E3USER_ROLE, только если её ещё нет
  select count(*) into v_count from dba_roles where role = 'E3USER_ROLE';
  if v_count = 0 then
    execute immediate 'CREATE ROLE E3USER_ROLE';
    dbms_output.put_line('Role E3USER_ROLE created.');
  else
    dbms_output.put_line('Role E3USER_ROLE already exists.');
  end if;

  -- Создаём роль E3WRITER_ROLE, только если её ещё нет
  select count(*) into v_count from dba_roles where role = 'E3WRITER_ROLE';
  if v_count = 0 then
    execute immediate 'CREATE ROLE E3WRITER_ROLE';
    dbms_output.put_line('Role E3WRITER_ROLE created.');
  else
    dbms_output.put_line('Role E3WRITER_ROLE already exists.');
  end if;
end;
/

COMMIT;