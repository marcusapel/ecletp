import os
import json
from urllib import request, parse
from dataclasses import dataclass
from typing import Optional
@dataclass
class Config:
    rddms_host: str
    data_partition_id: str
    dataspace_uri: str
    token: Optional[str]
    inline_threshold: int = 0
    compress_over_bytes: int = 1024
    array_path_style: str = 'uuid-path'
    use_protocol9: bool = True
    object_format: str = 'XML'
    epsg: int = 2334  # Default EPSG code
    @classmethod
    def from_env(cls) -> 'Config':
        return cls(
            rddms_host=os.getenv('RDDMS_HOST', os.getenv('ETP_URL', 'wss://equinordev.energy.azure.com/api/reservoir-ddms-etp/v2/')),
            data_partition_id=os.getenv('DATA_PARTITION_ID', os.getenv('DATA_PARTITION', 'data')),
            dataspace_uri=os.getenv('DATASPACE', "eml:///dataspace('maap/m25test')"),
            token=get_token(),
        )

def get_token() -> str:
    url = "https://login.microsoftonline.com/3aa4a235-b6e2-48d5-9195-7fcf05b459b0/oauth2/v2.0/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "grant_type": "refresh_token",
        "client_id": "ebd2bfee-ecba-47b7-a33c-017d0131879d",
        "scope": "7daee810-3f78-40c4-84c2-7a199428de18/.default openid offline_access",
        "refresh_token": os.environ['refresh_token']
    }
    try:
        payload = json.loads(
            request.urlopen(
                request.Request(url, data=parse.urlencode(data).encode("utf-8"), headers=headers, method="POST"),
                timeout=60
            ).read().decode("utf-8")
        )
        token = payload.get("access_token")
        if not token:
            raise RuntimeError(payload.get("error_description") or payload.get("error") or "No access_token in response")
        return token
    except Exception as e:
        raise RuntimeError(f"Token request failed: {e}")
