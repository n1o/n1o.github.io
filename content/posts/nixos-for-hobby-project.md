---
title: "Nixos for Hobby Project"
date: 2023-09-13T13:09:10+02:00
draft: true
---

Lately I started a side project: [CodeBreakers](https://codebreakers.re/courses/). It is nothing fancy just a site where I plan to release videos and articles about my latest passions Reverse Engineering and Binary Exploitation.

Building a website is nothing too fancy, and I did it a couple of times already. The only challange was deciding where and how to host it? My initial toughts were to use [Fly](https://fly.io/) (which is a great PaaS product where a lot of things just works), but I went instead to [Hetzner](hetzner.com/) and rented a virtual server!

Manging a virtual server is definetly a low level solution (and there is a lot of pain involved), but it gave me some space to explore the idea of using [NixOS](https://nixos.org/) on my server.

# Nixos
I used NixOS, as my main driver for arround 2 years, and oh boy, It did hurt on multiple occasions. For those that do not know, NixOS is a special linux ditribution build arround the [Nix](https://nix.dev/tutorials/first-steps/nix-language) programming language. What makes it special is that you have your whole operating system configuration as code. 

You may ask: Operating system as code sound a bit like Ansible, Puppet, Saltstack, ... Right? 

Well it depends, the techonologies mentioned above are way more sofisticated and offer more features. With NixOS this configuration is more similar to things like Terraform. Esentially you have a set of features that you can enable and configure. (There is also support for custom modules thus the limit is the sky.)


# My App
Ok so I have server that runs NixOS (if you don't maybe you can [Infect](https://github.com/elitak/nixos-infect)). What I want is:

1. Ngnix
2. Postgresql
3. Django

At Hetzner I got for a bit less than 6 Euros a virtual sever with 2 vCPUs and 4 GB of RAM and a fixed IP. This is beefy enough to run everything I want on a single machine (And if I need I can cost efficiently scale verticaly). By running on a single machine all my services can talk trough unix sockets, and the only open ports I need are:

```nix
networking.firewall = {
    enable = true;
    allowedTCPPorts = [22 80 443];
  };
``` 

NixOS is by default Secure thus All the ports that you want to open needs to be explicitly defined.

## Ngnix

I use Lets Encrypt certificates (want to have it as cheap as possible).

```nix
services.nginx = {
    enable = true;
    virtualHosts = {
	    "codebreakers.re" = {
	        addSSL = true;
	        enableACME = true;
	        locations."/.well-known/acme-challenge" = {
	        root = "/var/lib/acme/.challenges";
	        };
	    locations."/static/" = {
	    	alias = "/www/codebreakers/staticfiles/";
            extraConfig  = ''
            expires 1h;
            add_header Cache-Control "public";
            '';
        };
	    locations."/" = {
	    	proxyPass = "http://unix:/run/gunicorn.sock";
	    };
	  };
	};
};

users.users.nginx.extraGroups = [ "acme" ];
security.acme = {
    acceptTerms = true;
    email = "my@email.com";
};
```
Under the hood there is a an ```nginx``` linux user created under which Nginx runs as a systemd daemon. We do not have an ```ngnix.conf``` file but instead our configuration is inlined. 

## Postgresql

```nix
services.postgresql = {
  enable = true;
  package = pkgs.postgresql_15;
  ensureDatabases = [ "codebreakers" ];
  authentication = pkgs.lib.mkOverride 10 ''
    local all       all     trust
  '';

  initialScript = pkgs.writeText "backend-initScript" ''
    CREATE USER codebreakers WITH PASSWORD 'password';
    GRANT ALL PRIVILEGES ON DATABASE codebreakers TO codebreakers;
  '';
};
```

Similarly as for Ngnix under the hood there is a ```postgres``` user created. By default this does not open the typical ```5432``` port and wont allow TCP connections, but we wont need it since we can talk to it trough a unix socket located at ```/var/run/postgresql/.s.PGSQL.5432```


## Django

Now this was one of my pain points. In the past I did a lot of Python Machine Learning Projects on NixOS ad it was anything but friedly. Thus at the end I decided to run my application in a docker container (Technicaly it is [Podman](https://podman.io/))

```nix
{

virtualisation.oci-containers.containers = {
   djnago = {
	image = "ghcr.io/n1o/codebreakers-re:$TAG";
	entrypoint = "poetry";
	cmd = [
		"run" 
		"gunicorn" 
		"codebreakers.wsgi" 
		"-w" "2" 
		"-b" "unix:/run/gunicorn.sock"
		"--access-logfile" "'-'" 
		"--error-logfile" "'-' "
		];

	login = {
		username = "n1o";
		passwordFile = "/etc/nixos/github-container-password.txt";

	};
    user = "django";

	volumes = [
		"/var/run/postgresql/:/var/run/postgresql/"
		"/run/:/run/"
	];
   };
};
}
```

This tells Nixos to run django container as a systemd daemon.


# Release or CI/CD

I store all my code in a single repository at Github. I have github action that runs every time I create a new git Tag. This action collects static Django assets, builds my Docker Image that is pushed to Githubs Container Repository, replace ```$TAG``` above for the right value and copy the nixos configuration files trough SCP to the server. As the last step on the server it executes ```nixos-rebuild switch```

