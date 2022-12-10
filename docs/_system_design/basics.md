---
layout: page
title: System Design Basics
---


# Scalability

What is it that we really mean by scalability? 

A service is said to be scalable if when we increase the resources in a system, it results in increased performance in a manner proportional to resources added. 
Increasing performance in general means serving more units of work, but it can also be to handle larger units of work, such as when datasets grow.

In distributed systems there are other reasons for adding resources to a system; for example to improve the reliability of the offered service. Introducing redundancy is an important first line of defense against failures. An always-on service is said to be scalable if adding resources to facilitate redundancy does not result in a loss of performance.

Source [All Things Distributed](https://www.allthingsdistributed.com/2006/03/a_word_on_scalability.html)

### Horizontal vs Vertical Scaling

Horizontal scaling means that you are adding more servers -- cloning server.

Vertical scaling means that you are adding more power to your existing server -- adding CPU, RAM, storage etc. 

![Alt text](https://www.cloudzero.com/hubfs/blog/horizontal-vs-vertical-scaling.webp)


#### Advantages of horizontal scaling

* Scaling is easier from a hardware perspective - All horizontal scaling requires you to do is add additional machines to your current pool. It eliminates the need to analyze which system specifications you need to upgrade.
* Fewer periods of downtime - Because you’re adding a machine, you don’t have to switch the old machine off while scaling. If done effectively, there may never be a need for downtime and clients are less likely to be impacted.
* Increased resilience and fault tolerance - Relying on a single node for all your data and operations puts you at a high risk of losing it all when it fails. Distributing it among several nodes saves you from losing it all. 
* Increased performance - If you are using horizontal scaling to manage your network traffic, it allows for more endpoints for connections, considering that the load will be delegated among multiple machines.     

#### Disadvantages of horizontal scaling

* Increased complexity of maintenance and operation - Multiple servers are harder to maintain than a single server is. Additionally, you will need to add software for load balancing and possibly virtualization. Backing up your machines may also become a little more complex. You will need to ensure that nodes synchronize and communicate effectively. 
* Increased Initial costs - Adding new servers is far more expensive than upgrading old ones.   

#### Advantages of vertical scaling

* Cost-effective - Upgrading a pre-existing server costs less than purchasing a new one. Additionally, you are less likely to add new backup and virtualization software when scaling vertically. Maintenance costs may potentially remain the same too.
* Less complex process communication - When a single node handles all the layers of your services, it will not have to synchronize and communicate with other machines to work. This may result in faster responses.
* Less complicated maintenance - Not only is maintenance cheaper but it is less complex because of the number of nodes you will need to manage. 
* Less need for software changes - You are less likely to change how the software on a server works or how it is implemented.         

#### Disadvantages of vertical scaling

* Higher possibility for downtime - Unless you have a backup server that can handle operations and requests, you will need some considerable downtime to upgrade your machine. 
* Single point of failure - Having all your operations on a single server increases the risk of losing all your data if a hardware or software failure was to occur. 
* Upgrade limitations - There is a limitation to how much you can upgrade a machine. Every machine has its threshold for RAM, storage, and processing power. 

Source [CLOUDZERO](https://www.cloudzero.com/blog/horizontal-vs-vertical-scaling)