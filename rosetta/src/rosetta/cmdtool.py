#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from .support import *


def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false', '1', '0','y','n','yes','no','on','off']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False, '1':True, '0':False,'y':True,'n':False, 'yes':True,'no':False, 'on':True,'of':False}[s.lower()]



def add_boolean_argument(parser, name, default=False, dest=None, help=None):
    """Add a boolean argument to an ArgumentParser instance."""

    try:
      parser.add_argument('--' + name, action=argparse.BooleanOptionalAction,default=default,help = help)
      return 
    except: 
      pass

    # Fallback for Python < 3.9
    destname = dest or name.replace('-','_')

    onhelptext = None
    offhelptext = None
    if help is not None:
        onhepltext = help + (" (default)" if default else "")
        offhelptext = "Disable " + help + (" (default)" if default else "")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--' + name, dest=destname, action='store_true', help=onhelptext)
    group.add_argument('--no-' + name, dest=destname, action='store_false', help=offhelptext)
    defaults = {destname: default}
    parser.set_defaults(**defaults)


def get_boolean_cmdline(name, val):
    if val is None:
        return []
    if not val:
        return ['--no-'+name]
    return ['--'+name]



class ConfigParam:
  BOOL = NamedSentinel('BOOL')
  INT = NamedSentinel('INT')
  INTMIN = NamedSentinel('INTMIN')
  NUMBER = NamedSentinel('NUMBER')
  STRING = NamedSentinel('STRING')
  STRLIST = NamedSentinel('STRLIST')
  CMDARGLIST = NamedSentinel('CMDARGLIST')
  CMDARGDICT = NamedSentinel('CMDARGDICT')
  PATH = NamedSentinel('PATH')

  def __init__(self,parent,propname,type,cmdargname=None,inherits_from=[],defaultval=None,help=None,userparam=False):
    self.parent = parent
    self.propname = propname
    self.type = type
    self.cmdargname = cmdargname or propname
    self.defaultval = defaultval
    self.help = help
    self.userparam = userparam

    self.inherits = inherits_from
    self.val = None

    # For make walking easier
    self.mergeinherits = []



  def get_config_name(self):
    return self.propname


  def get_raw_value(self):
    # Is this the fastest way?
    return self.val


  def set_raw_value(self,val):
    assert val!=None, "Don't try to reset value this way"
    if self.type is ConfigParam.BOOL:
      val = bool(val)
    elif self.type in (ConfigParam.INT, ConfigParam.INTMIN):
        val = int (val)
    elif self.type is ConfigParam.NUMBER:
        if not isinstance(val,int):
            val = float(val)
    elif self.type is ConfigParam.STRING:
      val=str(val)
    elif self.type in (ConfigParam.STRLIST, ConfigParam.CMDARGLIST):
      val=list(val)
    elif self.type is ConfigParam.CMDARGDICT:
       val = dict(val)
    elif self.type is ConfigParam.PATH:
      val = mkpath(val)
    else :
      assert False, "Forgotten type implementation"

    self.val = val


  def get_effective_value(self,default=None):
    if self.type  in (ConfigParam.BOOL, ConfigParam.INT, ConfigParam.NUMBER, ConfigParam.STRING, ConfigParam.PATH):
      for prop in walk_inheritance_chain(self):
        # return first defined item
        val = prop.get_raw_value()
        if val != None:
          return val
    elif self.type is ConfigParam.INTMIN:
        l = [p.val for p in walk_inheritance_chain(self) if p.val != None]
        if not l :
            return None
        return min(l)
    elif self.type in (ConfigParam.STRLIST, ConfigParam.CMDARGLIST):
      return [item for items in (p.val for p in walk_inheritance_chain(self) if p.val != None) for item in items]
    elif self.type is (ConfigParam.CMDARGDICT):
        result = {}
        for d in walk_inheritance_chain(self):
            if d.val == None:
                continue
            for k,v in d.val.items():
              if k in result:
                  continue
              result[k] = v
        return result
    else:
      assert False, "Forgotten to implement?"
    return default


  def get_value(self,effective=True):
    if effective:
      return self.get_effective_value()
    else:
      return self.get_raw_value()


  def add_parser_argument(self,parser,propprefix='',cmdargprefix=''):
    name = '--' + cmdargprefix + self.cmdargname
    dest = propprefix + self.propname
    if self.type is ConfigParam.BOOL:
      add_boolean_argument(parser, cmdargprefix + self.cmdargname, default=self.defaultval, dest=dest,help=self.help)
    elif self.type in [ConfigParam.INT, ConfigParam.INTMIN ]:
      parser.add_argument(name,dest=dest,type=int,help=self.help)
    elif self.type is ConfigParam.NUMBER:
      parser.add_argument(name,dest=dest,type=float,help=self.help)
    elif self.type is ConfigParam.STRING:
      parser.add_argument(name,dest=dest,type=str,help=self.help)
    elif self.type is ConfigParam.STRLIST:
      parser.add_argument(name,dest=dest,type=str,help=self.help,action='append')
    elif self.type in (ConfigParam.CMDARGLIST, ConfigParam.CMDARGDICT):
      pluralpropname  = dest
      singularpropname =  propprefix + self.propname[:-1]
      pluralcmdargname = name
      singularcmdargname = '--' + cmdargprefix + self.cmdargname[:-1]
      parser.add_argument(pluralcmdargname,dest=pluralpropname,help=self.help,action='append')
      parser.add_argument(singularcmdargname,dest=singularpropname,help=self.help,action='append')
    elif self.type is ConfigParam.PATH:
      parser.add_argument(name,dest=dest,help=self.help,type=str)
    else:
      assert False,"forgotten to implement this for type?"

  def set_from_argdict(self,argdict,propprefix='',cmdargprefix=''):
    dest =  propprefix + self.propname
    lookup = argdict.get(dest)
    if self.type is ConfigParam.BOOL:
       if lookup ==None:
         return
       assert isinstance(lookup,bool)
    elif self.type in ( ConfigParam.INT, ConfigParam.INTMIN):
       if lookup ==None:
         return
       assert isinstance(lookup,int)
    elif self.type is ConfigParam.NUMBER:
       if lookup ==None:
         return
       assert isinstance(lookup,int) or isinstance(lookup,float)
    elif self.type is ConfigParam.STRING:
      if lookup ==None:
         return
      assert isinstance(lookup,str)
    elif self.type is ConfigParam.STRLIST:
      if lookup ==None:
         return
      assert isinstance(lookup,list)
    elif self.type is ConfigParam.CMDARGLIST:
      singularpropname =  propprefix + self.propname[:-1]
      plurallookup = lookup
      singularlookup = argdict.get(singularpropname)
      if plurallookup == None and singularlookup==None:
        return

      args = []
      if plurallookup != None:
        args += [arg for args in plurallookup for arg in shsplit(args)]
      if singularlookup != None:
        args += singularlookup
      lookup = args
    elif self.type is ConfigParam.CMDARGDICT:
      singularpropname =  propprefix + self.propname[:-1]
      plurallookup = lookup
      singularlookup = argdict.get(singularpropname)
      if plurallookup == None and singularlookup==None:
        return

      args = []
      if plurallookup != None:
        args += [arg for args in plurallookup for arg in shsplit(args)]
      if singularlookup != None:
        args += singularlookup
      lookup = {}
      for arg in args:
          k,v = arg.split('=',   maxsplit=1)
          lookup[k] = v
    elif self.type is ConfigParam.PATH:
      if lookup ==None:
         return
      assert isinstance(lookup,str)
      lookup = pathlib.Path(lookup)
    else:
      assert False,"forgotten to implement this for type?"

    self.set_raw_value(lookup)


  def get_cmdline(self,effective=False,cmdargprefix=''):
    cmdargname = cmdargprefix +  self.cmdargname

    val = self.get_value(effective=effective)
    if val==None:
      return []
    if self.type is ConfigParam.BOOL:
       return  get_boolean_cmdline(cmdargname, val)
    elif self.type in[ ConfigParam.STRING, ConfigParam.INT, ConfigParam.INTMIN]:
      return ['--' + cmdargname + '=' + str(val)]
    elif self.type in[ ConfigParam.STRLIST,ConfigParam.CMDARGLIST]:
      singularname = cmdargname[:-1]
      return [ '--' + singularname + '=' + v for v in val]
    elif self.type is ConfigParam.CMDARGDICT:
        singularname = cmdargname[:-1]
        result = []
        for k,v in val.items():
            result.append('--' + singularname)
            result.append(k + '=' + v)
        return result
    elif self.type is ConfigParam.PATH:
      assert isinstance(val,pathlib.Path)
      return ['--'+cmdargname +'=' + str(val)]
    else:
      assert False,"forgotten to implement this for type?"


class Configurable:
  parent = None
  propname = None

  def __init__(self,*args,**kwargs):
    super().__init__(*args,**kwargs)

    object.__setattr__(self,'props',{})
    object.__setattr__(self,'subconfigs',{})

    object.__setattr__(self,'mergeinherits',[])


  def __getattr__(self, name):
    props =  super().__getattribute__('props')
    if name in props:
      return props[name].get_effective_value()
    subconfigs =  super().__getattribute__('subconfigs')
    if name in self.subconfigs:
      return subconfigs[name]
    #return super().__getattr__(name)
    raise AttributeError



  def __setattr__(self, name, value):
    if name in self.props:
      self.props[name].set_raw_value(self,value)
      return
    assert not name in self.subconfigs,"Don't assign entire subconfigs"
    return super().__setattr__(name, value)


  def getprop(self,propname):
    return self.props[propname]

  # Would have been nice to make this declarative
  def addprop(self,propname,type,cmdargname=None,defaultval=None,inherits_from=[]):
    assert propname not in self.props
    assert propname not in self.subconfigs
    super().__setattr__('_' + propname, None)
    n = ConfigParam(self,propname,type,cmdargname=cmdargname,defaultval=defaultval,inherits_from=inherits_from)
    self.props[propname] = n
    return n




  def add_subconfig(self,propname,configurable):
    assert propname not in self.props
    assert propname not in self.subconfigs
    assert self!=configurable
    assert configurable.parent ==  None
    configurable.parent = self
    configurable.propname = propname
    self.subconfigs[propname] = configurable


  def add_parser_argument(self,parser,propprefix='',cmdargprefix=''):
    for k,p in self.props.items():
      p.add_parser_argument(parser,propprefix=propprefix,cmdargprefix=cmdargprefix)

    for k,s in self.subconfigs.items():
      subpropprefix =  ( propprefix or '') + k + '_'
      subcmdargprefix =  ( cmdargprefix or '') + k + '-'
      s.add_parser_argument(parser,propprefix=subpropprefix,cmdargprefix=subcmdargprefix)


  def create_parser(self):
    parser = argparse.ArgumentParser(allow_abbrev=False,add_help=False)
    self.add_parser_argument(parser)
    return parser


  def set_from_argdict(self,argdict,propprefix='',cmdargprefix=''):
    for k,p in self.props.items():
      p.set_from_argdict(argdict,propprefix=propprefix,cmdargprefix=cmdargprefix)

    for k,s in self.subconfigs.items():
      subpropprefix =  ( propprefix or '') + k + '_'
      subcmdargprefix =  ( cmdargprefix or '') + k + '-'
      s.set_from_argdict(argdict,propprefix=propprefix,cmdargprefix=cmdargprefix)

  def get_cmdline(self,effective=False,cmdargprefix=''):
    result = []
    for k,p in self.props.items():
       result .extend(p.get_cmdline(effective=effective,cmdargprefix=cmdargprefix))

    for k,s in self.subconfigs.items():
      subcmdargprefix =  ( cmdargprefix or '') + k + '-'
      result.extend(s.get_cmdline(effective=effective,cmdargprefix=cmdargprefix))
    return result


  @classmethod
  def from_argv(cls,argv):
    result = cls()
    parser = result.create_parser()
    argdict = parser.parse_args(argv)
    argdict = vars(argdict)
    result.set_from_argdict(argdict)
    return result

  @classmethod
  def from_merged(cls,*args):
    result = cls()
    result.mergeinherits = args
    return result


def follow_configurable_path(obj,path):
    cur = obj
    for elt in path:
      if elt in cur.props:
        # Avoid calling get_effective_value
        cur = cur.props[elt]
      else:
        cur = getattr(cur,elt)
    return cur

def walk_mergeprops(rootprop):
    cur = rootprop
    path = []
    while True:
      for mergeinherit in cur.mergeinherits:
        yield follow_configurable_path(mergeinherit,reversed(path))

      propname  = cur.propname
      parent = cur.parent

      if parent == None:
        # This was the root
        break

      path.append(propname)
      cur = parent



def walk_inheritance_chain(rootprop):
    worklist_propinherits = []
    worklist_mergeinherits = []
    visited_propinherits = set()
    visited_mergeinherits = set()

    def process_mergeinherits(seed=None):
      if seed != None:
        worklist_mergeinherits.append(seed)

      # Visit merge inheritances in dfs
      while worklist_mergeinherits:
        pair = worklist_mergeinherits.pop()
        if pair in visited_mergeinherits:
          continue
        visited_mergeinherits.add(pair)

        yield pair

        for subo in walk_mergeprops(pair):
          worklist_mergeinherits.append(subo)
          worklist_propinherits.append(subo)


    def process_propinherits(seed=None):
      if seed != None:
        worklist_propinherits.append(seed)

      # Visit prop inheritances in dfs
      while worklist_propinherits:
        obj = worklist_propinherits.pop()
        if obj in visited_propinherits:
          continue
        visited_propinherits.add(obj)

        # Visit merge inheritances first
        yield from process_mergeinherits(obj)

        # Push prop inheritances on top of the stack, so they are processed before merge-inheritance's prop inheritances
        worklist_propinherits.extend(obj.inherits)

    yield from process_propinherits(rootprop)




