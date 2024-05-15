import psycopg2
import csv

select_data_sql_1 = """
SELECT
    words.id AS word_id,
    words.body AS word_body,
    words.created_at AS word_created_at,
    words.user_id AS word_user_id,
    translations.id AS translation_id,
    translations.body AS translation_body,
    translations.rating,
    translations.created_at AS translation_created_at,
    translations.user_id AS translation_user_id,
    words.translations_count
FROM
    words
LEFT JOIN
    translations ON words.id = translations.word_id
ORDER BY
    words.id ASC,
    translations.created_at ASC;
"""

select_data_sql_2 = """
SELECT * FROM votes;
"""

def export_to_csv(sql_command,
                  filename,
                  user,
                  dbname):
    '''
    Connect to the database and export the data to a CSV 
    file after executing the SQL command.
    '''
    conn = psycopg2.connect(
        user=user,
        dbname=dbname
    )
    cursor = conn.cursor()
    cursor.execute(sql_command)
    rows = cursor.fetchall()
    with open(filename, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([desc[0] for desc in cursor.description])
        csv_writer.writerows(rows)
    cursor.close()
    conn.close()

# translations
export_to_csv(select_data_sql_1, 
              "../data/words_translations.csv", 
              "alexeykoshevoy", 
              "slovotvir")
# votes
export_to_csv(select_data_sql_2, 
              "../data/votes.csv", 
              "alexeykoshevoy", 
              "slovotvir")
