from ib_insync import IB
import random

def check_ibkr_connection(host='127.0.0.1', port=7497, client_id=2, timeout=5):
    """
    Tente de se connecter à TWS. Retourne True si la connexion réussit, sinon False.
    """
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=timeout)
        ib.disconnect()
        return True
    except Exception:
        return False


def get_ib_connection(host='127.0.0.1', port=7497, client_id=99, timeout=5):
    """
    Renvoie une instance IB connectée. Lève une exception si la connexion échoue.
    """
    ib = IB()
    try:
        client_id = random.randint(100, 999)  # évite les conflits
        ib.connect('127.0.0.1', 7497, clientId=client_id)
    except Exception as e:
        print(f"❌ Erreur de connexion TWS : {e}")
        raise
    return ib
