
echo LITELLM_MASTER_KEY="litellm-$(uuidgen)" > litellm.env 
echo LITELLM_SALT_KEY="litellm-$(uuidgen)" >> litellm.env 

echo DATABASE_URL="sqlite+aiosqlite:///./litellm.db" >> litellm.env
