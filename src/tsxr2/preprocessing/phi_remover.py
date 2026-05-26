"""PHI (Protected Health Information) removal utilities.

Implements HIPAA Safe Harbor de-identification by removing/anonymizing
protected health information from DICOM datasets before cloud transmission.
"""

from copy import deepcopy

from pydicom.dataset import Dataset

# DICOM tags containing PHI per HIPAA Safe Harbor method
# Reference: DICOM PS3.15 Annex E - Attribute Confidentiality Profiles
# Tags to completely remove (not needed for clinical context)
PHI_TAGS_TO_REMOVE = [
    # Patient details beyond basic identifiers
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientIDsSequence",
    "OtherPatientNames",
    # Physician contact info
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",
    "RequestingPhysician",
    "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy",
    "OperatorsName",
    # Institution details
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "StationName",
    # Dates that could identify
    "InstanceCreationDate",
    "InstanceCreationTime",
    # Device identifiers
    "DeviceSerialNumber",
    "PlateID",
    "CassetteID",
    "GeneratorID",
    # Procedure identifiers
    "ScheduledProcedureStepID",
    "RequestedProcedureID",
    "FillerOrderNumberImagingServiceRequest",
    "PlacerOrderNumberImagingServiceRequest",
]

# Tags to anonymize (set to placeholder rather than delete)
# These are kept with placeholder values to maintain DICOM structure validity
PHI_TAGS_TO_ANONYMIZE = {
    "PatientName": "",
    "PatientID": "ANONYMOUS",
    "AccessionNumber": "",
    "ReferringPhysicianName": "",
    "InstitutionName": "",
}


def remove_phi(dataset: Dataset, keep_dates: bool = False) -> Dataset:
    """Remove PHI from a DICOM dataset for HIPAA compliance.

    Creates a deep copy of the dataset and removes/anonymizes PHI tags
    to ensure the original dataset is not modified.

    Args:
        dataset: The DICOM dataset to anonymize.
        keep_dates: If True, preserves StudyDate/SeriesDate for clinical context.

    Returns:
        A new anonymized Dataset with PHI removed.
    """
    # Work on a copy to preserve the original
    anonymized = deepcopy(dataset)

    # First, anonymize tags that should have placeholder values
    # (preserves DICOM structure validity)
    for tag_name, placeholder in PHI_TAGS_TO_ANONYMIZE.items():
        if hasattr(anonymized, tag_name):
            setattr(anonymized, tag_name, placeholder)

    # Then, remove PHI tags that aren't needed
    for tag_name in PHI_TAGS_TO_REMOVE:
        if hasattr(anonymized, tag_name):
            delattr(anonymized, tag_name)

    # Optionally remove dates
    if not keep_dates:
        date_tags = ["StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"]
        time_tags = ["StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime"]
        for tag_name in date_tags + time_tags:
            if hasattr(anonymized, tag_name):
                delattr(anonymized, tag_name)

    return anonymized


def get_phi_fields(dataset: Dataset) -> dict[str, str]:
    """Extract PHI fields from a dataset for audit logging.

    This function identifies what PHI exists in a dataset before anonymization.
    Useful for audit trails and compliance verification.

    Args:
        dataset: The DICOM dataset to inspect.

    Returns:
        Dictionary mapping tag names to their values (as strings).
    """
    phi_found = {}

    all_phi_tags = PHI_TAGS_TO_REMOVE + list(PHI_TAGS_TO_ANONYMIZE.keys())
    for tag_name in all_phi_tags:
        if hasattr(dataset, tag_name):
            value = getattr(dataset, tag_name)
            if value:  # Only include non-empty values
                phi_found[tag_name] = str(value)

    return phi_found
