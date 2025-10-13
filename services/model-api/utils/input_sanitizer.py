"""
Input Sanitization Utilities
Provides comprehensive input validation and sanitization for security
"""

import re
import logging
from typing import List, Dict, Any, Optional
try:
    from bleach import clean
except ImportError:
    # Fallback if bleach is not available
    def clean(text, tags=None, strip=True):
        return text
import html

logger = logging.getLogger(__name__)

class InputSanitizer:
    """Handles input sanitization and validation"""
    
    def __init__(self):
        # Dangerous patterns for prompt injection
        self.dangerous_patterns = [
            r'ignore\s+(previous|all|system)\s+instructions?',
            r'forget\s+(everything|all|previous)',
            r'you\s+are\s+now\s+(dan|jailbroken|unrestricted)',
            r'system\s+prompt\s+(extraction|leak)',
            r'roleplay\s+as\s+(admin|developer|hacker)',
            r'execute\s+(command|code|script)',
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'<link[^>]*>',
            r'<meta[^>]*>',
            r'<style[^>]*>.*?</style>',
            r'<form[^>]*>',
            r'<input[^>]*>',
            r'<textarea[^>]*>',
            r'<select[^>]*>',
            r'<option[^>]*>',
            r'<button[^>]*>',
            r'<a[^>]*href\s*=',
            r'<img[^>]*src\s*=',
            r'<video[^>]*>',
            r'<audio[^>]*>',
            r'<source[^>]*>',
            r'<track[^>]*>',
            r'<canvas[^>]*>',
            r'<svg[^>]*>',
            r'<math[^>]*>',
            r'<table[^>]*>',
            r'<tr[^>]*>',
            r'<td[^>]*>',
            r'<th[^>]*>',
            r'<thead[^>]*>',
            r'<tbody[^>]*>',
            r'<tfoot[^>]*>',
            r'<col[^>]*>',
            r'<colgroup[^>]*>',
            r'<caption[^>]*>',
            r'<fieldset[^>]*>',
            r'<legend[^>]*>',
            r'<label[^>]*>',
            r'<output[^>]*>',
            r'<progress[^>]*>',
            r'<meter[^>]*>',
            r'<details[^>]*>',
            r'<summary[^>]*>',
            r'<dialog[^>]*>',
            r'<menu[^>]*>',
            r'<menuitem[^>]*>',
            r'<command[^>]*>',
            r'<keygen[^>]*>',
            r'<base[^>]*>',
            r'<area[^>]*>',
            r'<map[^>]*>',
            r'<param[^>]*>',
            r'<applet[^>]*>',
            r'<bgsound[^>]*>',
            r'<blink[^>]*>',
            r'<marquee[^>]*>',
            r'<nobr[^>]*>',
            r'<noembed[^>]*>',
            r'<noframes[^>]*>',
            r'<noscript[^>]*>',
            r'<plaintext[^>]*>',
            r'<xmp[^>]*>',
            r'<listing[^>]*>',
            r'<pre[^>]*>',
            r'<code[^>]*>',
            r'<kbd[^>]*>',
            r'<samp[^>]*>',
            r'<var[^>]*>',
            r'<dfn[^>]*>',
            r'<cite[^>]*>',
            r'<q[^>]*>',
            r'<blockquote[^>]*>',
            r'<address[^>]*>',
            r'<div[^>]*>',
            r'<span[^>]*>',
            r'<p[^>]*>',
            r'<h[1-6][^>]*>',
            r'<ul[^>]*>',
            r'<ol[^>]*>',
            r'<li[^>]*>',
            r'<dl[^>]*>',
            r'<dt[^>]*>',
            r'<dd[^>]*>',
            r'<dir[^>]*>',
            r'<menu[^>]*>',
            r'<hr[^>]*>',
            r'<br[^>]*>',
            r'<wbr[^>]*>',
            r'<b[^>]*>',
            r'<i[^>]*>',
            r'<u[^>]*>',
            r'<s[^>]*>',
            r'<strike[^>]*>',
            r'<del[^>]*>',
            r'<ins[^>]*>',
            r'<em[^>]*>',
            r'<strong[^>]*>',
            r'<small[^>]*>',
            r'<big[^>]*>',
            r'<sub[^>]*>',
            r'<sup[^>]*>',
            r'<tt[^>]*>',
            r'<font[^>]*>',
            r'<basefont[^>]*>',
            r'<center[^>]*>',
            r'<isindex[^>]*>',
            r'<nextid[^>]*>',
            r'<spacer[^>]*>',
            r'<multicol[^>]*>',
            r'<layer[^>]*>',
            r'<ilayer[^>]*>',
            r'<nobr[^>]*>',
            r'<wbr[^>]*>',
            r'<ruby[^>]*>',
            r'<rt[^>]*>',
            r'<rp[^>]*>',
            r'<bdo[^>]*>',
            r'<bdi[^>]*>',
            r'<mark[^>]*>',
            r'<time[^>]*>',
            r'<data[^>]*>',
            r'<article[^>]*>',
            r'<aside[^>]*>',
            r'<footer[^>]*>',
            r'<header[^>]*>',
            r'<main[^>]*>',
            r'<nav[^>]*>',
            r'<section[^>]*>',
            r'<figure[^>]*>',
            r'<figcaption[^>]*>',
            r'<hgroup[^>]*>',
            r'<abbr[^>]*>',
            r'<acronym[^>]*>',
            r'<b[^>]*>',
            r'<bdi[^>]*>',
            r'<bdo[^>]*>',
            r'<big[^>]*>',
            r'<br[^>]*>',
            r'<cite[^>]*>',
            r'<code[^>]*>',
            r'<del[^>]*>',
            r'<dfn[^>]*>',
            r'<em[^>]*>',
            r'<i[^>]*>',
            r'<ins[^>]*>',
            r'<kbd[^>]*>',
            r'<mark[^>]*>',
            r'<meter[^>]*>',
            r'<pre[^>]*>',
            r'<progress[^>]*>',
            r'<q[^>]*>',
            r'<ruby[^>]*>',
            r'<s[^>]*>',
            r'<samp[^>]*>',
            r'<small[^>]*>',
            r'<span[^>]*>',
            r'<strong[^>]*>',
            r'<sub[^>]*>',
            r'<sup[^>]*>',
            r'<time[^>]*>',
            r'<u[^>]*>',
            r'<var[^>]*>',
            r'<wbr[^>]*>',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.dangerous_patterns]
    
    def sanitize_text(self, text: str, max_length: int = 10000) -> str:
        """
        Sanitize and validate text input
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
            
        Raises:
            ValueError: If input is invalid or dangerous
        """
        if not text:
            raise ValueError("Empty text input")
        
        # Check length
        if len(text) > max_length:
            raise ValueError(f"Text input too long (max {max_length} characters)")
        
        # Strip whitespace
        text = text.strip()
        
        if not text:
            raise ValueError("Empty text after sanitization")
        
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                logger.warning(f"🚨 Dangerous pattern detected: {pattern.pattern}")
                raise ValueError("Input contains potentially dangerous content")
        
        # HTML sanitization
        text = clean(text, tags=[], strip=True)
        
        # Additional XSS protection
        text = html.escape(text, quote=True)
        
        # Validate charset (only allow safe characters)
        if not re.match(r'^[\w\s\.\,\!\?\-\:\;\(\)\[\]\{\}\"\'\/\\\@\#\$\%\^\&\*\+\=\<\>\|`~]+$', text):
            logger.warning("🚨 Invalid characters detected in input")
            raise ValueError("Input contains invalid characters")
        
        return text
    
    def validate_model_names(self, model_names: list) -> list:
        """
        Validate model names
        
        Args:
            model_names: List of model names to validate
            
        Returns:
            Validated model names
            
        Raises:
            ValueError: If model names are invalid
        """
        if not model_names:
            return []
        
        valid_models = []
        for model_name in model_names:
            if not isinstance(model_name, str):
                raise ValueError(f"Model name must be string: {model_name}")
            
            # Check for valid model name pattern
            if not re.match(r'^[a-zA-Z0-9_-]+$', model_name):
                raise ValueError(f"Invalid model name format: {model_name}")
            
            # Check length
            if len(model_name) > 100:
                raise ValueError(f"Model name too long: {model_name}")
            
            valid_models.append(model_name)
        
        return valid_models
    
    def sanitize_prediction_request(self, text: str, models: list = None, 
                                  ensemble: bool = False, max_length: int = 10000) -> dict:
        """
        Sanitize a complete prediction request
        
        Args:
            text: Input text
            models: List of model names
            ensemble: Whether to use ensemble
            max_length: Maximum text length
            
        Returns:
            Sanitized request data
            
        Raises:
            ValueError: If request is invalid
        """
        # Sanitize text
        sanitized_text = self.sanitize_text(text, max_length)
        
        # Validate models
        sanitized_models = self.validate_model_names(models or [])
        
        # Validate ensemble flag
        if not isinstance(ensemble, bool):
            raise ValueError("Ensemble flag must be boolean")
        
        return {
            "text": sanitized_text,
            "models": sanitized_models,
            "ensemble": ensemble
        }

# Global sanitizer instance
input_sanitizer = InputSanitizer()
