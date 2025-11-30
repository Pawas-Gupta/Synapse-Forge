"""Rate limiting utilities"""
import time
from collections import defaultdict
from typing import Dict, Tuple
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter
    
    Tracks requests per IP and enforces limits.
    """
    
    def __init__(self):
        """Initialize rate limiter"""
        self.requests: Dict[str, list] = defaultdict(list)
        self.limits = {
            'global': (100, 3600),  # 100 requests per hour
            '/apr
ite_rate_limturn    re
 teLimiter()iter = Ra _rate_lim   ne:
    er is Noe_limit_rat   if imiter
 rate_lglobal _   r"""
 ite limlobal rateor create gt "Ge
    ""teLimiter:) -> Rae_limiter(atet_re


def gNone_limiter = 
_ratnstanceer i limitate
# Global r   }

    ts
 ': self.limi     'limitss),
       questen(self.rents': lal_endpoi   'tot        eys())),
 f.requests.k k in sel[0] for':')k.split(': len(set(l_ips 'tota     
      urn {ret
        cs"""ter statistiate limi""Get r
        ") -> dict:(selfget_stats def  
   }
          1
 t - uncoquest_it - reg': limnin    'remai     1,
    ount + request_crent':ur'c      w,
      ': windo   'window       limit,
   'limit':            {
 urn True,et   r  
         )
  t_timed(currenpen].aprequests[keylf.
        selow request# Al         
             }
er
      retry_after': etry_aft'r        ,
        equest_countrent': r   'cur             : window,
ow'ind 'w          it,
     ': lim 'limit              False, {
      return ')
       dpoint}enp} on {ded: {ieeit exce limning(f'Ratogger.war       l)
     ])][0keylf.requests[ se -rent_time(curdow - iner = int(wft_atry          re  = limit:
uest_count >     if req      
   ts[key])
  f.requesn(selcount = le request_
       imit # Check l
       ]
                w
 windoime <- req_time _trent  if cur         
 y]equests[ke.rtime in selfime for req_req_t        
    [key] = [.requests self    uests
   d req ol   # Clean    
     )
    lobal'].limits['gelfnt, set(endpoimits.g self.li window =it,lim    nt
    t for endpoi# Get limi   
           ()
  .time= timeme nt_ti     curret}"
   :{endpoiny = f"{ip}ke      """
  )
        ictinfo_d, owedis_allTuple of (         turns:
         Re  
        endpoint
nt: API  endpoi   
         IP addressp: Client  i
          gs:     Ar     
   llowed
   est is a if requ Check"
               ""
bool, dict]: -> Tuple[global')int: str = 'str, endpo, ip: selfis_allowed(  def  
  ized')
   tial iniRateLimiter(' logger.info }
           inute
    per m# 50),   60g': (50,i/catalo  '/ap    nute
       mi),  # 10 per(10, 60/analyze': fpi/r