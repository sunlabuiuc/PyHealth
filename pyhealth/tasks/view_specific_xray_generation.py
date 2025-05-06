def view_specific_xray_generation(patients):
    """
    Task function to generate samples for view-specific X-ray generation.
    For each patient visit with multiple views, select a frontal view (PA or AP)
    as input and a lateral view (LL) as the target.
    """
    print("DEBUG: Entering view_specific_xray_generation")
    print(f"DEBUG: Input patients: {patients}")
    
    samples = []
    
    # Handle both cases: full patients dict or single patient dict
    if isinstance(patients, dict) and "patient_id" in patients:
        # Single patient dict (PyHealth's set_task behavior)
        patient_dict = {patients["patient_id"]: patients}
    else:
        # Full patients dict (our overridden set_task behavior)
        patient_dict = patients
    
    for patient_id, patient in patient_dict.items():
        print(f"DEBUG: Processing patient {patient_id}")
        for visit in patient.get("visits", []):
            print(f"DEBUG: Processing visit {visit.get('visit_id')}")
            # Get all events (X-rays) in this visit
            events = visit.get("events", [])
            if len(events) < 2:  # Need at least two views for input-target pair
                continue
                
            # Find frontal (PA or AP) and lateral (LL) views
            frontal_event = None
            lateral_event = None
            for event in events:
                view_position = event.get("view_position")
                if view_position in ["PA", "AP"]:
                    frontal_event = event
                elif view_position == "LL":
                    lateral_event = event
            
            # Create sample if both views are available
            if frontal_event and lateral_event:
                sample = {
                    "patient_id": patient_id,
                    "visit_id": visit.get("visit_id"),
                    "input_front_view": frontal_event.get("image_path"),
                    "input_view_position": frontal_event.get("view_position"),
                    "target_view": lateral_event.get("view_position"),
                    "target_path": lateral_event.get("image_path"),
                }
                samples.append(sample)
    
    print(f"DEBUG: Generated samples: {samples}")
    return samples