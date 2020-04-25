import sys
import mysql.connector
from mysql.connector import errorcode

            
def connect_db():
    """Attempt to connect to GCP MySQL database and return connectin"""   
    try:                                                                    # To connect to database in Ubuntu VM:
        cnx = mysql.connector.connect(user='root', password='malware',      # user=malware
                                    host='34.87.119.210',                   # 10.255.252.177 
                                    database='malware-analysis')            # malwareanalysis
        print("CONNECTION SUCCESS")

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
            print(sys.stderr, "does not exist")
        else:
            print(err)
            print(sys.stderr, "does not exist")
        sys.exit(1)

    return cnx


def insert_technique(tech_id, t_name, text, source, date_crawled, m_id):
    """
    Insert ATT&CK technique into database

    Parameters:
        tech_id (str) -- Technique ID (e.g. T1034)
        t_name (str) -- Technique name
        text (str) -- Sentence technique was found in
        source (str) -- URL of text's source
        date_crawled (str) -- Current date
        m_id (int) -- malware's id as defined in malwares table
        
    """
    # We will open and close cnx and close for each insert
    cnx = connect_db()
    cursor = cnx.cursor()

    # Replace unwanted/error-inducing characters
    text = text.replace('\n', ' ')
    text = text.replace('\'', '')
    text = text.replace('\"', '')

    # Try inserting data
    try:
        print("INSERTING INTO DATABASE")
        command = f"INSERT INTO techniques (tech_id, technique, text, source, date_crawled, malwares_m_id) \
                    VALUES ('{tech_id}', '{t_name}', '{text}', '{source}', '{date_crawled}', {m_id});"
        cursor.execute(command)
        print("INSERT SUCCESS")
    except Exception as e:
        print('INSERT FAILED:', e)

    # Try committing changes to the database
    try:    
        cnx.commit()          
        print('COMMITTED')
    except Exception as e:
        print("COULD NOT COMMIT DATA:", e)

    cursor.close()
    cnx.close()
