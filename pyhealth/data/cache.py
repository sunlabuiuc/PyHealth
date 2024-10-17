import msgpack
import os
from typing import Dict, List, Any
from pyhealth.data.data_v2 import Patient, Event
from datetime import datetime
def patient_default(obj):
    if isinstance(obj, Patient):
        return {
            "__patient__": True,
            "patient_id": obj.patient_id,
            "birth_datetime": obj.birth_datetime.isoformat() if obj.birth_datetime else None,
            "death_datetime": obj.death_datetime.isoformat() if obj.death_datetime else None,
            "gender": obj.gender,
            "ethnicity": obj.ethnicity,
            "attr_dict": obj.attr_dict,
            "events": [event_default(event) for event in obj.events]
        }
    elif isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def event_default(obj):
    if isinstance(obj, Event):
        return {
            "__event__": True,
            "code": obj.code,
            "table": obj.table,
            "vocabulary": obj.vocabulary,
            "visit_id": obj.visit_id,
            "patient_id": obj.patient_id,
            "timestamp": obj.timestamp.isoformat() if obj.timestamp else None,
            "item_id": obj.item_id,
            "attr_dict": obj.attr_dict
        }
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def patient_hook(obj):
    if "__patient__" in obj:
        patient = Patient(
            patient_id=obj["patient_id"],
            birth_datetime=datetime.fromisoformat(obj["birth_datetime"]) if obj["birth_datetime"] else None,
            death_datetime=datetime.fromisoformat(obj["death_datetime"]) if obj["death_datetime"] else None,
            gender=obj["gender"],
            ethnicity=obj["ethnicity"]
        )
        patient.attr_dict = obj["attr_dict"]
        for event_data in obj["events"]:
            patient.add_event(event_hook(event_data))
        return patient
    return obj

def event_hook(obj):
    if "__event__" in obj:
        event = Event(
            code=obj["code"],
            table=obj["table"],
            vocabulary=obj["vocabulary"],
            visit_id=obj["visit_id"],
            patient_id=obj["patient_id"],
            timestamp=datetime.fromisoformat(obj["timestamp"]) if obj["timestamp"] else None,
            item_id=obj["item_id"]
        )
        event.attr_dict = obj["attr_dict"]
        return event
    return obj

def write_msgpack_patients(data: Dict[str, Patient], filepath: str):
    """
    Write a dictionary of Patient objects to a MessagePack file.
    
    Args:
    data (Dict[str, Patient]): Dictionary with patient IDs as keys and Patient objects as values.
    filepath (str): Path to the file where data will be written.
    """
    with open(filepath, 'wb') as f:
        msgpack.pack(data, f, default=patient_default)

def read_msgpack_patients(filepath: str) -> Dict[str, Patient]:
    """
    Read a dictionary of Patient objects from a MessagePack file.
    
    Args:
    filepath (str): Path to the file to read from.
    
    Returns:
    Dict[str, Patient]: Dictionary with patient IDs as keys and Patient objects as values.
    """
    with open(filepath, 'rb') as f:
        data = msgpack.unpack(f, object_hook=patient_hook)
    return data


def write_msgpack(data: Dict[str, Any], filepath: str) -> None:
    """
    Write a dictionary to a MessagePack file.
    
    Args:
    data (Dict[str, Any]): The dictionary to be written.
    filepath (str): The path to the file where data will be written.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Open the file in binary write mode
    with open(filepath, "wb") as f:
        # Pack and write the data
        packed = msgpack.packb(data, use_bin_type=True)
        f.write(packed)

def read_msgpack(filepath: str) -> Dict[str, Any]:
    """
    Read a dictionary from a MessagePack file.
    
    Args:
    filepath (str): The path to the file to be read.
    
    Returns:
    Dict[str, Any]: The dictionary read from the file.
    """
    # Open the file in binary read mode
    with open(filepath, "rb") as f:
        # Read and unpack the data
        packed = f.read()
        unpacked = msgpack.unpackb(packed, raw=False)
    
    return unpacked